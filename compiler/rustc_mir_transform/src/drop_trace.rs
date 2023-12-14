use rustc_hir::LangItem;
use rustc_middle::mir::visit::MutVisitor;
use rustc_middle::mir::{
    BasicBlockData, CallSource, Const, ConstOperand, LocalDecl, MirPass, Operand, Place,
    Terminator, TerminatorKind,
};
use rustc_middle::ty::{InstanceDef, List, Ty, TyCtxt};
use rustc_span::{sym, DUMMY_SP};

pub struct AddDropTraces;

impl<'tcx> MirPass<'tcx> for AddDropTraces {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut rustc_middle::mir::Body<'tcx>) {
        if let InstanceDef::Item(item) = body.source.instance {
            debug!("AddDropTraces({})", tcx.def_path_str(item));

            if tcx.coroutine_is_async(item) {
                debug!("{} is async", tcx.def_path_str(item));
            }

            if tcx.has_attr(item, sym::leak) {
                struct DropIntoGoto<'tcx> {
                    tcx: TyCtxt<'tcx>,
                }
                impl<'tcx> MutVisitor<'tcx> for DropIntoGoto<'tcx> {
                    fn tcx<'a>(&'a self) -> TyCtxt<'tcx> {
                        self.tcx
                    }

                    fn visit_terminator(
                        &mut self,
                        terminator: &mut rustc_middle::mir::Terminator<'tcx>,
                        location: rustc_middle::mir::Location,
                    ) {
                        if let TerminatorKind::Drop { target, .. } = terminator.kind {
                            terminator.kind = TerminatorKind::Goto { target }
                        }
                        self.super_terminator(terminator, location)
                    }
                }

                DropIntoGoto { tcx }.visit_body(body);
            } else if let Some(attr) = tcx.get_attr(item, sym::trace_drop) {
                debug!("AddDropTraces({})", tcx.def_path_str(item));

                let destination = *body.trace_drop_destination.get_or_insert_with(|| Place {
                    local: body.local_decls.push(LocalDecl::new(Ty::new_unit(tcx), DUMMY_SP)),
                    projection: List::empty(),
                });
                let trace_drop_def_id = tcx.require_lang_item(LangItem::TraceDrop, Some(attr.span));

                let blocks_with_drop: Vec<_> = body
                    .basic_blocks_mut()
                    .iter_enumerated()
                    .filter_map(|(block, data)| {
                        if let TerminatorKind::Drop { unwind, place, .. } = data.terminator().kind {
                            Some((block, unwind, place.local))
                        } else {
                            None
                        }
                    })
                    .collect();

                for &(predrop_block, unwind, droped_local) in &blocks_with_drop {
                    let predrop_block_data = &mut body.basic_blocks_mut()[predrop_block];
                    let drop_block_data = BasicBlockData {
                        statements: vec![],
                        terminator: predrop_block_data.terminator.take(),
                        is_cleanup: predrop_block_data.is_cleanup,
                    };
                    let trace_drop_func = Operand::Constant(Box::new(ConstOperand {
                        span: DUMMY_SP,
                        user_ty: None,
                        const_: Const::zero_sized(Ty::new_fn_def(
                            tcx,
                            trace_drop_def_id,
                            [body.local_decls[droped_local].ty],
                        )),
                    }));
                    let drop_block = body.basic_blocks_mut().push(drop_block_data);
                    let drop_block_data = &body.basic_blocks_mut()[drop_block];
                    let terminator = Terminator {
                        source_info: drop_block_data.terminator().source_info,
                        kind: TerminatorKind::Call {
                            func: trace_drop_func.clone(),
                            args: vec![],
                            destination,
                            target: Some(drop_block),
                            unwind,
                            call_source: CallSource::Misc,
                            fn_span: DUMMY_SP,
                        },
                    };
                    body.basic_blocks_mut()[predrop_block].terminator = Some(terminator);
                }
            }
        }
    }
}
