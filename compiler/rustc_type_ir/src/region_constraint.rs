use derive_where::derive_where;
use indexmap::IndexSet;
#[cfg(feature = "nightly")]
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::transitive_relation::{TransitiveRelation, TransitiveRelationBuilder};
use tracing::{debug, instrument};

use crate::data_structures::IndexMap;
use crate::fold::TypeSuperFoldable;
use crate::inherent::*;
use crate::relate::{Relate, RelateResult, TypeRelation, VarianceDiagInfo};
use crate::visit::TypeSuperVisitable;
use crate::{
    AliasTy, Binder, BoundRegion, BoundVar, BoundVariableKind, ConstKind, DebruijnIndex,
    FallibleTypeFolder, InferCtxtLike, InferTy, Interner, OutlivesPredicate, RegionKind, TyKind,
    TypeFoldable, TypeFolder, TypeVisitable, TypeVisitor, TypingMode, UniverseIndex, Variance,
    VisitorResult,
};

#[derive_where(Clone, Debug; I: Interner)]
pub struct Assumptions<I: Interner> {
    pub type_outlives: Vec<Binder<I, OutlivesPredicate<I, I::Ty>>>,
    pub region_outlives: TransitiveRelation<I::Region>,
    pub inverse_region_outlives: TransitiveRelation<I::Region>,
}

impl<I: Interner> Assumptions<I> {
    pub fn empty() -> Self {
        Self {
            type_outlives: Vec::new(),
            region_outlives: TransitiveRelationBuilder::default().freeze(),
            inverse_region_outlives: TransitiveRelationBuilder::default().freeze(),
        }
    }

    pub fn new(
        type_outlives: Vec<Binder<I, OutlivesPredicate<I, I::Ty>>>,
        region_outlives: TransitiveRelation<I::Region>,
    ) -> Self {
        Self {
            inverse_region_outlives: {
                let mut builder = TransitiveRelationBuilder::default();
                for (r1, r2) in region_outlives.base_edges() {
                    builder.add(r2, r1);
                }
                builder.freeze()
            },
            type_outlives,
            region_outlives,
        }
    }
}

#[derive_where(Clone, Hash, PartialEq, Debug; I: Interner)]
pub enum RegionConstraint<I: Interner> {
    Ambiguity,
    RegionOutlives(I::Region, I::Region),
    AliasTyOutlivesFromEnv(Binder<I, (AliasTy<I>, I::Region)>),
    /// This is an `I::Ty` for two reasons:
    /// 1. We need the type visitable impl to be able to `visit_ty` on this so canonicalization
    ///    knows about the placeholder
    /// 2. When exiting the trait solver there may be placeholder outlives corresponding to params
    ///    from the root universe. These need to be changed from a `Placeholder` to the original
    ///    `Param`.
    PlaceholderTyOutlives(I::Ty, I::Region),

    And(Box<[RegionConstraint<I>]>),
    Or(Box<[RegionConstraint<I>]>),
}

#[cfg(feature = "nightly")]
// This is not a derived impl because a perfect derive leads to cycle errors which
// means the trait is never actually implemented but the compiler doesn't tell you
// that so if you get a *WEIRD* error where its just telling you random types don't
// implement HashStable.... it's because of that
impl<CTX, I: Interner> HashStable<CTX> for RegionConstraint<I>
where
    I::Region: HashStable<CTX>,
    AliasTy<I>: HashStable<CTX>,
    I::Ty: HashStable<CTX>,
    I::BoundVarKinds: HashStable<CTX>,
{
    #[inline]
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        use RegionConstraint::*;

        std::mem::discriminant(self).hash_stable(hcx, hasher);
        match self {
            Ambiguity => (),
            RegionOutlives(a, b) => {
                a.hash_stable(hcx, hasher);
                b.hash_stable(hcx, hasher);
            }
            AliasTyOutlivesFromEnv(outlives) => {
                outlives.hash_stable(hcx, hasher);
            }
            PlaceholderTyOutlives(a, b) => {
                a.hash_stable(hcx, hasher);
                b.hash_stable(hcx, hasher);
            }
            And(and) => {
                for a in and.iter() {
                    a.hash_stable(hcx, hasher);
                }
            }
            Or(or) => {
                for a in or.iter() {
                    a.hash_stable(hcx, hasher);
                }
            }
        }
    }
}

impl<I: Interner> TypeFoldable<I> for RegionConstraint<I> {
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, f: &mut F) -> Result<Self, F::Error> {
        use RegionConstraint::*;
        Ok(match self {
            Ambiguity => self,
            RegionOutlives(a, b) => RegionOutlives(a.try_fold_with(f)?, b.try_fold_with(f)?),
            AliasTyOutlivesFromEnv(outlives) => AliasTyOutlivesFromEnv(outlives.try_fold_with(f)?),
            PlaceholderTyOutlives(a, b) => {
                PlaceholderTyOutlives(a.try_fold_with(f)?, b.try_fold_with(f)?)
            }
            And(and) => {
                let mut new_and = Vec::new();
                for a in and {
                    new_and.push(a.try_fold_with(f)?);
                }
                And(new_and.into_boxed_slice())
            }
            Or(or) => {
                let mut new_or = Vec::new();
                for a in or {
                    new_or.push(a.try_fold_with(f)?);
                }
                Or(new_or.into_boxed_slice())
            }
        })
    }

    fn fold_with<F: TypeFolder<I>>(self, f: &mut F) -> Self {
        use RegionConstraint::*;
        match self {
            Ambiguity => self,
            RegionOutlives(a, b) => RegionOutlives(a.fold_with(f), b.fold_with(f)),
            AliasTyOutlivesFromEnv(outlives) => AliasTyOutlivesFromEnv(outlives.fold_with(f)),
            PlaceholderTyOutlives(a, b) => PlaceholderTyOutlives(a.fold_with(f), b.fold_with(f)),
            And(and) => {
                let mut new_and = Vec::new();
                for a in and {
                    new_and.push(a.fold_with(f));
                }
                And(new_and.into_boxed_slice())
            }
            Or(or) => {
                let mut new_or = Vec::new();
                for a in or {
                    new_or.push(a.fold_with(f));
                }
                Or(new_or.into_boxed_slice())
            }
        }
    }
}

impl<I: Interner> TypeVisitable<I> for RegionConstraint<I> {
    fn visit_with<F: TypeVisitor<I>>(&self, f: &mut F) -> F::Result {
        use core::ops::ControlFlow::*;

        use RegionConstraint::*;

        match self {
            Ambiguity => (),
            RegionOutlives(a, b) => {
                if let b @ Break(_) = a.visit_with(f).branch() {
                    return F::Result::from_branch(b);
                };
                if let b @ Break(_) = b.visit_with(f).branch() {
                    return F::Result::from_branch(b);
                };
            }
            AliasTyOutlivesFromEnv(outlives) => {
                return outlives.visit_with(f);
            }
            PlaceholderTyOutlives(a, b) => {
                if let b @ Break(_) = a.visit_with(f).branch() {
                    return F::Result::from_branch(b);
                };
                if let b @ Break(_) = b.visit_with(f).branch() {
                    return F::Result::from_branch(b);
                };
            }
            And(and) => {
                for a in and {
                    if let b @ Break(_) = a.visit_with(f).branch() {
                        return F::Result::from_branch(b);
                    };
                }
            }
            Or(or) => {
                for a in or {
                    if let b @ Break(_) = a.visit_with(f).branch() {
                        return F::Result::from_branch(b);
                    };
                }
            }
        };

        F::Result::output()
    }
}

impl<I: Interner> Default for RegionConstraint<I> {
    fn default() -> Self {
        Self::new_true()
    }
}

impl<I: Interner> RegionConstraint<I> {
    pub fn new_true() -> Self {
        RegionConstraint::And(Box::new([]))
    }

    pub fn is_true(&self) -> bool {
        match self {
            Self::And(and) => and.is_empty(),
            _ => false,
        }
    }

    pub fn new_false() -> Self {
        RegionConstraint::Or(Box::new([]))
    }

    pub fn is_false(&self) -> bool {
        match self {
            Self::Or(or) => or.is_empty(),
            _ => false,
        }
    }

    pub fn is_or(&self) -> bool {
        matches!(self, Self::Or(_))
    }

    pub fn is_ambig(&self) -> bool {
        matches!(self, Self::Ambiguity)
    }
}

pub fn max_universe<Infcx: InferCtxtLike<Interner = I>, I: Interner, T: TypeVisitable<I>>(
    infcx: &Infcx,
    t: T,
) -> UniverseIndex {
    let mut visitor = MaxUniverse::new(infcx);
    t.visit_with(&mut visitor);
    visitor.max_universe()
}

struct MaxUniverse<'a, Infcx: InferCtxtLike> {
    max_universe: UniverseIndex,
    infcx: &'a Infcx,
}

impl<'a, Infcx: InferCtxtLike> MaxUniverse<'a, Infcx> {
    fn new(infcx: &'a Infcx) -> Self {
        MaxUniverse { infcx, max_universe: UniverseIndex::ROOT }
    }

    fn max_universe(self) -> UniverseIndex {
        self.max_universe
    }
}

impl<'a, Infcx: InferCtxtLike<Interner = I>, I: Interner> TypeVisitor<I>
    for MaxUniverse<'a, Infcx>
{
    fn visit_ty(&mut self, t: I::Ty) {
        if let TyKind::Placeholder(placeholder) = t.kind() {
            self.max_universe = self.max_universe.max(placeholder.universe);
        }

        if let TyKind::Infer(InferTy::TyVar(inf)) = t.kind() {
            let u = self.infcx.universe_of_ty(inf).unwrap();
            debug!("var {inf:?} in universe {u:?}");
            self.max_universe = self.max_universe.max(u)
        }

        t.super_visit_with(self)
    }

    fn visit_const(&mut self, c: I::Const) {
        if let ConstKind::Placeholder(placeholder) = c.kind() {
            self.max_universe = self.max_universe.max(placeholder.universe);
        }

        if let ConstKind::Infer(rustc_type_ir::InferConst::Var(inf)) = c.kind() {
            let u = self.infcx.universe_of_ct(inf).unwrap();
            debug!("var {inf:?} in universe {u:?}");
            self.max_universe = self.max_universe.max(u)
        }

        c.super_visit_with(self)
    }

    fn visit_region(&mut self, r: I::Region) {
        if let RegionKind::RePlaceholder(placeholder) = r.kind() {
            self.max_universe = self.max_universe.max(placeholder.universe);
        }

        if let RegionKind::ReVar(var) = r.kind() {
            let u = self.infcx.universe_of_lt(var).unwrap();
            debug!("var {var:?} in universe {u:?}");
            self.max_universe = self.max_universe.max(u)
        }
    }
}
