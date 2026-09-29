
use montyformat::chess::Attacks;
use crate::sydposition::{PieceType, Position};

#[derive(Clone)]
pub struct SydFilter{
    pub min_depth: u8,
    // bound condition
    pub require_exact: bool,
    // bestmove conditions
    pub require_bm: bool,
    pub filter_check: bool,
    pub filter_tactical: bool,
    pub filter_castling: bool,

    // pieces condition
    pub min_pieces: u32,

    // eval condition
    pub max_eval: u16,
}

impl Default for SydFilter {
    fn default() -> Self {
        Self {
            min_depth: 5,
            require_exact: false,
            require_bm: false,
            filter_check: true,
            filter_tactical: true,
            filter_castling: false,
            min_pieces: 4,
            max_eval: 20000,
        }
    }
}

impl SydFilter {
    pub fn nofilter() -> Self {
        Self {
            min_depth: 0,
            require_exact: false,
            require_bm: false,
            filter_check: false,
            filter_tactical: false,
            filter_castling: false,
            min_pieces: 0,
            max_eval: u16::MAX,
        }
    }
    pub fn should_filter(
        &self,
        pos: &Position
    ) -> bool {
        let occ: u64 = pos.bbs[0] | pos.bbs[1];
        if pos.depth < self.min_depth {
            return true;
        }
        if self.require_exact && pos.bound != 0 {
            return true;
        }
        if self.require_bm && pos.bm == 0 {
            return true;
        }
        if pos.score.unsigned_abs() >= self.max_eval {
            return true;
        }
        if occ.count_ones() < self.min_pieces {
            return true;
        }
        if self.filter_castling && (pos.bm >> 14) == 2 {
            return true;
        }
        if self.filter_tactical && pos.bm != 0 {
            let to: u8 = ((pos.bm >> 6) & 0x3f) as u8;
            let flag: u8 = (pos.bm >> 14) as u8;
            if (occ & (1 << to as u64)) != 0 || flag == 3 || flag == 1 {
                return true;
            }
        }
        if self.filter_check {
            let kingpos = (pos.bbs[PieceType::King as usize + 2] & pos.bbs[pos.stm as usize]).trailing_ones();
            let attacks: u64 = 
                (
                    (Attacks::pawn(kingpos as usize, pos.stm as usize) & pos.bbs[2]) |
                    (Attacks::knight(kingpos as usize) & pos.bbs[3]) |
                    (Attacks::bishop(kingpos as usize, occ) & pos.bbs[4]) |
                    (Attacks::rook(kingpos as usize, occ) & pos.bbs[5]) |
                    (Attacks::queen(kingpos as usize, occ) & pos.bbs[6])
                ) & pos.bbs[pos.stm as usize ^ 1];
            if attacks != 0 {
                return true;
            }

        }
        return false;
    }
}