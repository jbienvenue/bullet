//use acyclib::{graph::builder::Affine, trainer::logger};
//use bullet_cuda_backend::CudaMarker;
use bullet_lib::{
    game::inputs::SparseInputType,
    nn::optimiser::{AdamW, AdamWParams},
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder},
};
use bullet_lib::value::loader::ViriBinpackLoader;
use bullet_lib::game::outputs::MaterialCount;

use crate::threat_inputs::ThreatInputs;
use viriformat::dataformat::Filter;

const L1_SIZE: usize = 512;
const L2_SIZE: usize = 16;
const L3_SIZE: usize = 32;

//const INPUT_BUCKETS: usize = 4;
const OUTPUT_BUCKETS: usize = 8;

#[rustfmt::skip]
const KING_BUCKET_LAYOUT: [usize; 32] = [
    0, 0, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
];

fn main() {
    //logger::set_cbcs(true);

    // hyperparams to fiddle with
    let initial_lr = 0.001;
    let final_lr = 0.001 * 0.3f32.powi(5);
    let superbatches: usize = 600;

    let inputs = ThreatInputs::new(KING_BUCKET_LAYOUT);
    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(inputs)
        .output_buckets(MaterialCount::<OUTPUT_BUCKETS>)
        .save_format(&[
            // merge in the factoriser weights
            SavedFormat::id("l0w"),
            SavedFormat::id("l0b"),
            SavedFormat::id("l1w"),
            SavedFormat::id("l1b"),
            SavedFormat::id("l2w"),
            SavedFormat::id("l2b"),
            SavedFormat::id("l3w"),
            SavedFormat::id("l3b"),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            
            let l0 = builder.new_affine("l0", inputs.num_inputs(), L1_SIZE);
            l0.init_with_effective_input_size(20000);

            // output layer weights
            let l1 = builder.new_affine("l1", L1_SIZE, OUTPUT_BUCKETS*L2_SIZE);
            let l2 = builder.new_affine("l2", L2_SIZE, OUTPUT_BUCKETS*L3_SIZE);
            let l3 = builder.new_affine("l3", L3_SIZE, OUTPUT_BUCKETS);

            // inference
            //let ft = |input, start, end| l0.slice(start, end).forward(input).crelu();
            //let stm_hidden = ft(stm_inputs, 0, L1_SIZE / 2) * ft(stm_inputs, L1_SIZE / 2, L1_SIZE);
            //let ntm_hidden = ft(ntm_inputs, 0, L1_SIZE / 2) * ft(ntm_inputs, L1_SIZE / 2, L1_SIZE);
            let stm_hidden = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm_hidden = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let hl1 = stm_hidden.concat(ntm_hidden);
            let hl2 = l1.forward(hl1).select(output_buckets).screlu();
            let hl3 = l2.forward(hl2).select(output_buckets).crelu();
            l3.forward(hl3).select(output_buckets)
        });

    // need to account for factoriser weight magnitudes
    let stricter_clipping = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", stricter_clipping);
    let settings = LocalSettings { threads: 10, test_set: None, output_directory: "checkpoints", batch_queue_size: 32 };

    let schedule = TrainingSchedule {
        net_id: "ti_4ib_512_16_32_dfrc_linWDLall".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384*8,
            batches_per_superbatch: 6104/8,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::LinearWDL{start: 0.3, end: 0.7},
        lr_scheduler: lr::CosineDecayLR { initial_lr, final_lr, final_superbatch: superbatches },
        save_rate: 10,
    };
    let data_loader = {
        let file_path = "data/data9-10-12-13-14-17-interleaved.vf";
        let buffer_size_mb = 4096;
        let threads = 20;
        let filter = Filter {
            min_ply: 8,
            ..Default::default()
        };
        ViriBinpackLoader::new(file_path, buffer_size_mb, threads, filter)
    };


    //trainer.load_from_checkpoint("checkpoints/ti_4ib_512_16_32_dfrc-260");

    trainer.run(&schedule, &settings, &data_loader);

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ] {
        let eval = trainer.eval(fen);
        println!("FEN:  {fen}");
        println!("EVAL: {}", 400.0 * eval);
    }
}

mod threat_inputs {
    use bullet_lib::game::{formats::bulletformat::ChessBoard, inputs};

    use montyformat::chess::{Attacks, Piece, Side};

    use crate::{offsets, threats::map_piece_threat};

    #[derive(Clone, Copy)]
    pub struct ThreatInputs {
        buckets: [usize; 64],
        total_features: usize,
    }

    impl ThreatInputs {
        pub const TOTAL_THREATS: usize = 2 * offsets::END;

        pub fn new(buckets: [usize; 32]) -> Self {
            let num_buckets = inputs::get_num_buckets(&buckets);

            let mut expanded = [0; 64];
            for (idx, elem) in expanded.iter_mut().enumerate() {
                *elem = buckets[(idx / 8) * 4 + [0, 1, 2, 3, 3, 2, 1, 0][idx % 8]];
            }

            let total_features = Self::TOTAL_THREATS + 768 * num_buckets + 768;

            Self { buckets: expanded, total_features }
        }
    }

    impl Default for ThreatInputs {
        fn default() -> Self {
            let total_features = Self::TOTAL_THREATS + 768 + 768;
            Self { buckets: [0; 64], total_features }
        }
    }

    impl inputs::SparseInputType for ThreatInputs {
        type RequiredDataType = ChessBoard;

        fn num_inputs(&self) -> usize {
            self.total_features
        }

        fn max_active(&self) -> usize {
            128 + 32
        }

        fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
            let get = |ksq| (if ksq % 8 > 3 { 7 } else { 0 }, 768 * self.buckets[usize::from(ksq)]);
            let (stm_flip, stm_bucket) = get(pos.our_ksq());
            let (ntm_flip, ntm_bucket) = get(pos.opp_ksq());

            #[rustfmt::skip]
            inputs::Chess768.map_features(pos, |stm, ntm| {
                f(
                    ThreatInputs::TOTAL_THREATS + stm ^ stm_flip,
                    ThreatInputs::TOTAL_THREATS + ntm ^ ntm_flip,
                );
                f(
                    ThreatInputs::TOTAL_THREATS + 768 + stm_bucket + (stm ^ stm_flip),
                    ThreatInputs::TOTAL_THREATS + 768 + ntm_bucket + (ntm ^ ntm_flip),
                );
            });

            let mut bbs = [0; 8];
            for (pc, sq) in pos.into_iter() {
                let pt = 2 + usize::from(pc & 7);
                let c = usize::from(pc & 8 > 0);
                let bit = 1 << sq;
                bbs[c] |= bit;
                bbs[pt] |= bit;
            }

            let mut stm_count = 0;
            let mut stm_feats = [0; 128];
            map_threat_features(bbs, |stm| {
                stm_feats[stm_count] = stm;
                stm_count += 1;
            });

            bbs.swap(0, 1);
            for bb in &mut bbs {
                *bb = bb.swap_bytes();
            }

            let mut ntm_count = 0;
            let mut ntm_feats = [0; 128];
            map_threat_features(bbs, |ntm| {
                ntm_feats[ntm_count] = ntm;
                ntm_count += 1;
            });

            assert_eq!(stm_count, ntm_count);

            for (&stm, &ntm) in stm_feats.iter().zip(ntm_feats.iter()).take(stm_count) {
                f(stm, ntm);
            }
        }

        fn shorthand(&self) -> String {
            todo!();
        }

        fn description(&self) -> String {
            todo!();
        }
    }

    fn map_bb<F: FnMut(usize)>(mut bb: u64, mut f: F) {
        while bb > 0 {
            let sq = bb.trailing_zeros() as usize;
            f(sq);
            bb &= bb - 1;
        }
    }

    fn flip_horizontal(mut bb: u64) -> u64 {
        const K1: u64 = 0x5555555555555555;
        const K2: u64 = 0x3333333333333333;
        const K4: u64 = 0x0f0f0f0f0f0f0f0f;
        bb = ((bb >> 1) & K1) | ((bb & K1) << 1);
        bb = ((bb >> 2) & K2) | ((bb & K2) << 2);
        ((bb >> 4) & K4) | ((bb & K4) << 4)
    }

    fn map_threat_features<F: FnMut(usize)>(mut bbs: [u64; 8], mut f: F) {
        // horiontal mirror
        let ksq = (bbs[0] & bbs[Piece::KING]).trailing_zeros();
        if ksq % 8 > 3 {
            for bb in bbs.iter_mut() {
                *bb = flip_horizontal(*bb);
            }
        };

        let mut pieces = [13; 64];
        for side in [Side::WHITE, Side::BLACK] {
            for piece in Piece::PAWN..=Piece::KING {
                let pc = 6 * side + piece - 2;
                map_bb(bbs[side] & bbs[piece], |sq| pieces[sq] = pc);
            }
        }

        let mut count = 0;

        let occ = bbs[0] | bbs[1];

        for side in [Side::WHITE, Side::BLACK] {
            let side_offset = offsets::END * side;
            let opps = bbs[side ^ 1];

            for piece in Piece::PAWN..Piece::KING {
                map_bb(bbs[side] & bbs[piece], |sq| {
                    let threats = match piece {
                        Piece::PAWN => Attacks::pawn(sq, side),
                        Piece::KNIGHT => Attacks::knight(sq),
                        Piece::BISHOP => Attacks::bishop(sq, occ),
                        Piece::ROOK => Attacks::rook(sq, occ),
                        Piece::QUEEN => Attacks::queen(sq, occ),
                        _ => unreachable!(),
                    } & occ;

                    count += 1;
                    map_bb(threats, |dest| {
                        let enemy = (1 << dest) & opps > 0;
                        if let Some(idx) = map_piece_threat(piece, sq, dest, pieces[dest], enemy) {
                            f(side_offset + idx);
                            count += 1;
                        }
                    });
                });
            }
        }
    }
}

mod threats {
    use montyformat::chess::Piece;

    use crate::{attacks, indices, offsets};

    pub fn map_piece_threat(piece: usize, src: usize, dest: usize, target: usize, enemy: bool) -> Option<usize> {
        match piece {
            Piece::PAWN => map_pawn_threat(src, dest, target, enemy),
            Piece::KNIGHT => map_knight_threat(src, dest, target),
            Piece::BISHOP => map_bishop_threat(src, dest, target),
            Piece::ROOK => map_rook_threat(src, dest, target),
            Piece::QUEEN => map_queen_threat(src, dest, target),
            Piece::KING => panic!(),
            _ => unreachable!(),
        }
    }

    fn below(src: usize, dest: usize, table: &[u64; 64]) -> usize {
        (table[src] & ((1 << dest) - 1)).count_ones() as usize
    }

    const fn offset_mapping<const N: usize>(a: [usize; N]) -> [usize; 12] {
        let mut res = [usize::MAX; 12];

        let mut i = 0;
        while i < N {
            res[a[i] - 2] = i;
            res[a[i] + 4] = i + N;
            i += 1;
        }

        res
    }

    fn target_is(target: usize, piece: usize) -> bool {
        target % 6 == piece - 2
    }

    fn map_pawn_threat(src: usize, dest: usize, target: usize, enemy: bool) -> Option<usize> {
        const MAP: [usize; 12] = offset_mapping([Piece::PAWN, Piece::KNIGHT, Piece::ROOK]);

        if MAP[target] == usize::MAX || (enemy && dest > src && target_is(target, Piece::PAWN)) {
            return None;
        }

        let id = if dest.abs_diff(src) == [9, 7][(dest > src) as usize] { 0 } else { 1 };
        let attack = 2 * (src % 8) + id - 1;
        let threat = offsets::PAWN + MAP[target] * indices::PAWN + (src / 8 - 1) * 14 + attack;
        Some(threat)
    }

    fn map_knight_threat(src: usize, dest: usize, target: usize) -> Option<usize> {
        const MAP: [usize; 12] = offset_mapping([Piece::PAWN, Piece::KNIGHT, Piece::BISHOP, Piece::ROOK, Piece::QUEEN]);

        if MAP[target] == usize::MAX || dest > src && target_is(target, Piece::KNIGHT) {
            return None;
        }

        let idx = indices::KNIGHT[src] + below(src, dest, &attacks::KNIGHT);
        let threat = offsets::KNIGHT + MAP[target] * indices::KNIGHT[64] + idx;
        Some(threat)
    }

    fn map_bishop_threat(src: usize, dest: usize, target: usize) -> Option<usize> {
        const MAP: [usize; 12] = offset_mapping([Piece::PAWN, Piece::KNIGHT, Piece::BISHOP, Piece::ROOK]);

        if MAP[target] == usize::MAX || dest > src && target_is(target, Piece::BISHOP) {
            return None;
        }

        let idx = indices::BISHOP[src] + below(src, dest, &attacks::BISHOP);
        let threat = offsets::BISHOP + MAP[target] * indices::BISHOP[64] + idx;
        Some(threat)
    }

    fn map_rook_threat(src: usize, dest: usize, target: usize) -> Option<usize> {
        const MAP: [usize; 12] = offset_mapping([Piece::PAWN, Piece::KNIGHT, Piece::BISHOP, Piece::ROOK]);

        if MAP[target] == usize::MAX || dest > src && target_is(target, Piece::ROOK) {
            return None;
        }

        let idx = indices::ROOK[src] + below(src, dest, &attacks::ROOK);
        let threat = offsets::ROOK + MAP[target] * indices::ROOK[64] + idx;
        Some(threat)
    }

    fn map_queen_threat(src: usize, dest: usize, target: usize) -> Option<usize> {
        const MAP: [usize; 12] = offset_mapping([Piece::PAWN, Piece::KNIGHT, Piece::BISHOP, Piece::ROOK, Piece::QUEEN]);

        if MAP[target] == usize::MAX || dest > src && target_is(target, Piece::QUEEN) {
            return None;
        }

        let idx = indices::QUEEN[src] + below(src, dest, &attacks::QUEEN);
        let threat = offsets::QUEEN + MAP[target] * indices::QUEEN[64] + idx;
        Some(threat)
    }
}

mod offsets {
    use super::indices;

    pub const PAWN: usize = 0;
    pub const KNIGHT: usize = PAWN + 6 * indices::PAWN;
    pub const BISHOP: usize = KNIGHT + 10 * indices::KNIGHT[64];
    pub const ROOK: usize = BISHOP + 8 * indices::BISHOP[64];
    pub const QUEEN: usize = ROOK + 8 * indices::ROOK[64];
    pub const END: usize = QUEEN + 10 * indices::QUEEN[64];
}

mod indices {
    use super::attacks;

    macro_rules! init_add_assign {
        (|$sq:ident, $init:expr, $size:literal | $($rest:tt)+) => {{
            let mut $sq = 0;
            let mut res = [{$($rest)+}; $size + 1];
            let mut val = $init;
            while $sq < $size {
                res[$sq] = val;
                val += {$($rest)+};
                $sq += 1;
            }

            res[$size] = val;

            res
        }};
    }

    pub const PAWN: usize = 84;
    pub const KNIGHT: [usize; 65] = init_add_assign!(|sq, 0, 64| attacks::KNIGHT[sq].count_ones() as usize);
    pub const BISHOP: [usize; 65] = init_add_assign!(|sq, 0, 64| attacks::BISHOP[sq].count_ones() as usize);
    pub const ROOK: [usize; 65] = init_add_assign!(|sq, 0, 64| attacks::ROOK[sq].count_ones() as usize);
    pub const QUEEN: [usize; 65] = init_add_assign!(|sq, 0, 64| attacks::QUEEN[sq].count_ones() as usize);
}

mod attacks {
    macro_rules! init {
        (|$sq:ident, $size:literal | $($rest:tt)+) => {{
            let mut $sq = 0;
            let mut res = [{$($rest)+}; $size];
            while $sq < $size {
                res[$sq] = {$($rest)+};
                $sq += 1;
            }
            res
        }};
    }

    const A: u64 = 0x0101_0101_0101_0101;
//    const H: u64 = A << 7;

    const DIAGS: [u64; 15] = [
        0x0100_0000_0000_0000,
        0x0201_0000_0000_0000,
        0x0402_0100_0000_0000,
        0x0804_0201_0000_0000,
        0x1008_0402_0100_0000,
        0x2010_0804_0201_0000,
        0x4020_1008_0402_0100,
        0x8040_2010_0804_0201,
        0x0080_4020_1008_0402,
        0x0000_8040_2010_0804,
        0x0000_0080_4020_1008,
        0x0000_0000_8040_2010,
        0x0000_0000_0080_4020,
        0x0000_0000_0000_8040,
        0x0000_0000_0000_0080,
    ];

    pub const KNIGHT: [u64; 64] = init!(|sq, 64| {
        let n = 1 << sq;
        let h1 = ((n >> 1) & 0x7f7f_7f7f_7f7f_7f7f) | ((n << 1) & 0xfefe_fefe_fefe_fefe);
        let h2 = ((n >> 2) & 0x3f3f_3f3f_3f3f_3f3f) | ((n << 2) & 0xfcfc_fcfc_fcfc_fcfc);
        (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8)
    });

    pub const BISHOP: [u64; 64] = init!(|sq, 64| {
        let rank = sq / 8;
        let file = sq % 8;
        DIAGS[file + rank].swap_bytes() ^ DIAGS[7 + file - rank]
    });

    pub const ROOK: [u64; 64] = init!(|sq, 64| {
        let rank = sq / 8;
        let file = sq % 8;
        (0xFF << (rank * 8)) ^ (A << file)
    });

    pub const QUEEN: [u64; 64] = init!(|sq, 64| BISHOP[sq] | ROOK[sq]);
}
