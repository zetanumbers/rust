use std::collections::VecDeque;
use std::iter;

use rustc_hir::LangItem;
use rustc_index::bit_set::BitSet;
use rustc_middle::mir::{
    self, BasicBlock, BasicBlockData, CallSource, ConstOperand, Local, LocalDecl, MirPass, Operand,
    Place, Rvalue, SourceInfo, Statement, StatementKind, SwitchTargets, Terminator, TerminatorKind,
    UnwindAction, START_BLOCK,
};
use rustc_middle::ty::util::Discr;
use rustc_middle::ty::{
    Const, GenericArg, InstanceDef, List, ParamEnv, Region, RegionKind, Ty, TyCtxt,
};
use rustc_span::source_map::dummy_spanned;
use rustc_span::DUMMY_SP;

pub struct AddAsyncDrop;

impl<'tcx> MirPass<'tcx> for AddAsyncDrop {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut rustc_middle::mir::Body<'tcx>) {
        let InstanceDef::Item(item) = body.source.instance else { return };
        if !tcx.coroutine_is_async(item) {
            return;
        }

        let param_env = tcx.param_env_reveal_all_normalized(item);
        let blocks_to_add_async_drop = get_blocks_to_add_async_drop(tcx, body, param_env);
        if blocks_to_add_async_drop.is_empty() {
            return;
        }
        debug!("AddAsyncDrop({})", tcx.def_path_str(item));

        let resume_arg = Place { local: Local::from_u32(2), projection: List::empty() };

        let item_span = tcx.span_of_impl(item).ok();

        let pin_new_unchecked_fn = tcx.require_lang_item(LangItem::PinNewUnchecked, item_span);

        let get_context_fn = tcx.require_lang_item(LangItem::GetContext, item_span);
        let get_context_fn = Ty::new_fn_def(
            tcx,
            get_context_fn,
            iter::repeat_with(|| Region::new_from_kind(tcx, RegionKind::ReErased)).take(2),
        );
        let context_ref_ty = Ty::new_task_context(tcx);
        let context_ref_place = Place {
            local: body.local_decls.push(LocalDecl::new(context_ref_ty, DUMMY_SP)),
            projection: List::empty(),
        };
        let get_context_fn = Operand::Constant(Box::new(ConstOperand {
            span: DUMMY_SP,
            user_ty: None,
            const_: mir::Const::zero_sized(get_context_fn),
        }));

        let poll_drop_fn = tcx.require_lang_item(LangItem::AsyncDropPoll, item_span);

        let poll_enum = tcx.adt_def(tcx.require_lang_item(LangItem::Poll, item_span));
        let Discr { val: poll_ready_discr, ty: poll_discr_ty } = poll_enum
            .discriminant_for_variant(
                tcx,
                poll_enum
                    .variant_index_with_id(tcx.require_lang_item(LangItem::PollReady, item_span)),
            );
        let poll_pending_discr = poll_enum
            .discriminant_for_variant(
                tcx,
                poll_enum
                    .variant_index_with_id(tcx.require_lang_item(LangItem::PollPending, item_span)),
            )
            .val;
        let poll_discr_place = Place {
            local: body.local_decls.push(LocalDecl::new(poll_discr_ty, DUMMY_SP)),
            projection: List::empty(),
        };

        let basic_blocks = body.basic_blocks.as_mut();
        let unreachable_block = basic_blocks.push(BasicBlockData::new(Some(Terminator {
            source_info: SourceInfo::outermost(body.span),
            kind: TerminatorKind::Unreachable,
        })));

        let unit_value = Operand::Constant(Box::new(ConstOperand {
            span: DUMMY_SP,
            user_ty: None,
            const_: mir::Const::zero_sized(Ty::new_unit(tcx)),
        }));

        let coroutine_drop_end_block = basic_blocks.push(BasicBlockData::new(Some(Terminator {
            source_info: SourceInfo::outermost(body.span),
            kind: TerminatorKind::CoroutineDrop,
        })));

        // Prepend poll loops before drops
        for old_drop_block in blocks_to_add_async_drop.iter() {
            let old_drop_block_data = &mut basic_blocks[old_drop_block];
            let drop_terminator = old_drop_block_data.terminator.take().unwrap();
            let Terminator {
                kind:
                    TerminatorKind::Drop {
                        place: drop_place,
                        unwind: drop_unwind,
                        replace: drop_replace,
                        target: _,
                    },
                source_info,
            } = drop_terminator
            else {
                bug!();
            };
            let drop_ty = drop_place.ty(&body.local_decls, tcx).ty;
            let drop_ref_ty =
                Ty::new_mut_ref(tcx, Region::new_from_kind(tcx, RegionKind::ReErased), drop_ty);
            let drop_ref_place = Place {
                local: body.local_decls.push(LocalDecl::with_source_info(drop_ref_ty, source_info)),
                projection: List::empty(),
            };

            let pin_new_unchecked_fn = Ty::new_fn_def(
                tcx,
                pin_new_unchecked_fn,
                // <Pointer, const HOST: bool>
                [GenericArg::from(drop_ref_ty), GenericArg::from(Const::from_bool(tcx, true))],
            );
            let pin_ty = pin_new_unchecked_fn.fn_sig(tcx).output().no_bound_vars().unwrap();
            let pin_place = Place {
                local: body.local_decls.push(LocalDecl::new(pin_ty, DUMMY_SP)),
                projection: List::empty(),
            };
            let pin_new_unchecked_fn = Operand::Constant(Box::new(ConstOperand {
                span: DUMMY_SP,
                user_ty: None,
                const_: mir::Const::zero_sized(pin_new_unchecked_fn),
            }));

            let poll_drop_fn =
                Ty::new_fn_def(tcx, poll_drop_fn, iter::once(GenericArg::from(drop_ty)));
            // TODO move to outer scope
            let poll_unit_ty = poll_drop_fn.fn_sig(tcx).output().no_bound_vars().unwrap();
            let poll_unit_place = Place {
                local: body.local_decls.push(LocalDecl::new(poll_unit_ty, DUMMY_SP)),
                projection: List::empty(),
            };
            let poll_drop_fn = Operand::Constant(Box::new(ConstOperand {
                span: DUMMY_SP,
                user_ty: None,
                const_: mir::Const::zero_sized(poll_drop_fn),
            }));

            let temporaries = [
                drop_ref_place.local,
                pin_place.local,
                context_ref_place.local,
                poll_unit_place.local,
                poll_discr_place.local,
            ];

            let pin_new_unchecked_block = basic_blocks.push(BasicBlockData::new(None));
            let get_context_block = basic_blocks.push(BasicBlockData::new(None));
            let poll_drop_block = basic_blocks.push(BasicBlockData::new(None));
            let switch_block = basic_blocks.push(BasicBlockData::new(None));
            let yield_block = basic_blocks.push(BasicBlockData::new(None));
            let resume_arg_move_back = basic_blocks.push(BasicBlockData::new(None));
            let drop_block = basic_blocks.push(BasicBlockData::new(Some(drop_terminator)));
            let coroutine_drop_begin_block = basic_blocks.push(BasicBlockData::new(None));

            {
                // Generating coroutine drop branch for async drop
                basic_blocks[coroutine_drop_begin_block].statements.extend(
                    temporaries.iter().copied().map(|local| Statement {
                        source_info,
                        kind: StatementKind::StorageDead(local),
                    }),
                );
                basic_blocks[coroutine_drop_begin_block].terminator = Some(Terminator {
                    source_info,
                    kind: TerminatorKind::Drop {
                        place: drop_place,
                        // FIXME: Other drops
                        target: coroutine_drop_end_block,
                        unwind: drop_unwind,
                        replace: drop_replace,
                    },
                });
            }

            let unwind_begin_block = basic_blocks.push(BasicBlockData {
                statements: temporaries
                    .iter()
                    .copied()
                    .map(|local| Statement { source_info, kind: StatementKind::StorageDead(local) })
                    .collect(),
                terminator: Some(Terminator {
                    source_info,
                    kind: match drop_unwind {
                        UnwindAction::Continue => TerminatorKind::UnwindResume,
                        UnwindAction::Unreachable => TerminatorKind::Unreachable,
                        UnwindAction::Terminate(reason) => TerminatorKind::UnwindTerminate(reason),
                        UnwindAction::Cleanup(target) => TerminatorKind::Goto { target },
                    },
                }),
                is_cleanup: true,
            });

            // Defining blocks
            basic_blocks[old_drop_block].statements.extend(
                temporaries.iter().copied().map(|local| Statement {
                    source_info,
                    kind: StatementKind::StorageLive(local),
                }),
            );
            basic_blocks[old_drop_block].terminator = Some(Terminator {
                source_info,
                kind: TerminatorKind::Goto { target: pin_new_unchecked_block },
            });
            basic_blocks[pin_new_unchecked_block].statements.push(Statement {
                source_info,
                kind: StatementKind::Assign(Box::new((
                    drop_ref_place,
                    Rvalue::Ref(
                        Region::new_from_kind(tcx, RegionKind::ReErased),
                        mir::BorrowKind::Mut { kind: mir::MutBorrowKind::Default },
                        drop_place,
                    ),
                ))),
            });
            basic_blocks[pin_new_unchecked_block].terminator = Some(Terminator {
                source_info,
                kind: TerminatorKind::Call {
                    func: pin_new_unchecked_fn,
                    args: vec![dummy_spanned(Operand::Move(drop_ref_place))],
                    destination: pin_place,
                    target: Some(get_context_block),
                    unwind: UnwindAction::Cleanup(unwind_begin_block),
                    call_source: CallSource::Misc,
                    fn_span: DUMMY_SP,
                },
            });
            let resume_arg_temp_arg = Place {
                local: body
                    .local_decls
                    .push(LocalDecl::new(resume_arg.ty(&body.local_decls, tcx).ty, DUMMY_SP)),
                projection: List::empty(),
            };
            basic_blocks[get_context_block].statements.push(Statement {
                source_info,
                kind: StatementKind::Assign(Box::new((
                    resume_arg_temp_arg,
                    Rvalue::Use(Operand::Move(resume_arg)),
                ))),
            });
            basic_blocks[get_context_block].terminator = Some(Terminator {
                source_info,
                kind: TerminatorKind::Call {
                    func: get_context_fn.clone(),
                    args: vec![dummy_spanned(Operand::Move(resume_arg_temp_arg))],
                    destination: context_ref_place,
                    target: Some(poll_drop_block),
                    unwind: UnwindAction::Cleanup(unwind_begin_block),
                    call_source: CallSource::Misc,
                    fn_span: DUMMY_SP,
                },
            });
            basic_blocks[poll_drop_block].terminator = Some(Terminator {
                source_info,
                kind: TerminatorKind::Call {
                    func: poll_drop_fn.clone(),
                    args: vec![
                        dummy_spanned(Operand::Move(pin_place)),
                        dummy_spanned(Operand::Move(context_ref_place)),
                    ],
                    destination: poll_unit_place,
                    target: Some(switch_block),
                    unwind: UnwindAction::Cleanup(unwind_begin_block),
                    call_source: CallSource::Misc,
                    fn_span: DUMMY_SP,
                },
            });
            basic_blocks[switch_block].statements.push(Statement {
                source_info,
                kind: StatementKind::Assign(Box::new((
                    poll_discr_place,
                    Rvalue::Discriminant(poll_unit_place),
                ))),
            });
            basic_blocks[switch_block].terminator = Some(Terminator {
                source_info,
                kind: TerminatorKind::SwitchInt {
                    discr: Operand::Move(poll_discr_place),
                    targets: SwitchTargets::new(
                        [(poll_ready_discr, drop_block), (poll_pending_discr, yield_block)]
                            .into_iter(),
                        unreachable_block,
                    ),
                },
            });
            let resume_arg_temp_dest = Place {
                local: body
                    .local_decls
                    .push(LocalDecl::new(resume_arg.ty(&body.local_decls, tcx).ty, DUMMY_SP)),
                projection: List::empty(),
            };
            basic_blocks[yield_block].terminator = Some(Terminator {
                source_info,
                kind: TerminatorKind::Yield {
                    value: unit_value.clone(),
                    resume: resume_arg_move_back,
                    resume_arg: resume_arg_temp_dest,
                    drop: Some(coroutine_drop_begin_block),
                },
            });
            basic_blocks[resume_arg_move_back].statements.push(Statement {
                source_info,
                kind: StatementKind::Assign(Box::new((
                    resume_arg,
                    Rvalue::Use(Operand::Move(resume_arg_temp_dest)),
                ))),
            });
            basic_blocks[resume_arg_move_back].terminator = Some(Terminator {
                source_info,
                kind: TerminatorKind::Goto { target: pin_new_unchecked_block },
            });
            basic_blocks[drop_block].statements.extend(
                temporaries.iter().copied().map(|local| Statement {
                    source_info,
                    kind: StatementKind::StorageDead(local),
                }),
            );
        }
    }
}

fn get_blocks_to_add_async_drop<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    param_env: ParamEnv<'tcx>,
) -> BitSet<BasicBlock> {
    // Walk through every block on happy async path
    let blocks_count = body.basic_blocks.len();
    let mut blocks_to_add_async_drop = BitSet::new_empty(blocks_count);
    let mut blocks_reached = BitSet::new_empty(blocks_count);
    let mut blocks_to_check = VecDeque::from([START_BLOCK]);
    while let Some(block) = blocks_to_check.pop_back() {
        match &body.basic_blocks[block].terminator().kind {
            TerminatorKind::Drop { target, place, unwind: _, replace: _ } => {
                if place.ty(body, tcx).ty.is_async_drop(tcx, param_env) {
                    blocks_to_add_async_drop.insert(block);
                }
                if blocks_reached.insert(*target) {
                    blocks_to_check.push_back(*target);
                }
            }

            TerminatorKind::Goto { target }
            | TerminatorKind::Assert { target, .. }
            | TerminatorKind::Yield { resume: target, .. }
            | TerminatorKind::Call { target: Some(target), .. }
            | TerminatorKind::InlineAsm { destination: Some(target), .. }
            | TerminatorKind::FalseUnwind { real_target: target, .. } => {
                if blocks_reached.insert(*target) {
                    blocks_to_check.push_back(*target);
                }
            }

            TerminatorKind::SwitchInt { targets, .. } => blocks_to_check.extend(
                targets.all_targets().iter().filter(|&target| blocks_reached.insert(*target)),
            ),

            TerminatorKind::FalseEdge { real_target, imaginary_target } => {
                if blocks_reached.insert(*real_target) {
                    blocks_to_check.push_back(*real_target);
                }
                if blocks_reached.insert(*imaginary_target) {
                    blocks_to_check.push_back(*imaginary_target);
                }
            }

            TerminatorKind::UnwindResume
            | TerminatorKind::UnwindTerminate(_)
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::Call { target: None, .. }
            | TerminatorKind::InlineAsm { destination: None, .. }
            | TerminatorKind::CoroutineDrop => (),
        }
    }
    blocks_to_add_async_drop
}
