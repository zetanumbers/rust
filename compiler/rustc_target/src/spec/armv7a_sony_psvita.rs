use crate::spec::{LinkerFlavor, LldFlavor, PanicStrategy, RelocModel, Target, TargetOptions};

// The PSVita has custom linker requirements.
const LINKER_SCRIPT: &str = include_str!("./armv7a_sony_psvita_linker_script.ld");

pub fn target() -> Target {
    let pre_link_args = TargetOptions::link_args(LinkerFlavor::Ld, &["--emit-relocs"]);

    Target {
        llvm_target: "armv7a-none-eabihf".into(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),

        options: TargetOptions {
            os: "psvita".into(),
            vendor: "sony".into(),
            linker_flavor: LinkerFlavor::Lld(LldFlavor::Ld),
            cpu: "cortex-a9".into(),
            abi: "eabihf".into(),
            max_atomic_width: Some(64),
            panic_strategy: PanicStrategy::Abort,
            executables: true,
            linker: Some("rust-lld".into()),
            relocation_model: RelocModel::DynamicNoPic,

            features: "+v7,+vfp3,-d32,+thumb2,+neon,+strict-align".into(),

            pre_link_args,
            link_script: Some(LINKER_SCRIPT.into()),
            ..Default::default()
        },
    }
}
