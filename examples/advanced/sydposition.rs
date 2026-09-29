use std::convert::TryFrom;
use bullet_lib::game::formats::bulletformat::ChessBoard;

#[repr(u8)]
#[allow(dead_code)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub enum PieceType{
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
    Castle,
    Void
}

#[repr(u8)]
#[allow(dead_code)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub enum Pieces{
    WPawn, BPawn,
    WKnight, BKnight,
    WBishop, BBishop,
    WRook, BRook,
    WQueen, BQueen,
    WKing, BKing,
    WCASTLE, BCASLTE,
    Void
}

#[allow(dead_code)]
pub enum Color{White, Black}

impl TryFrom<u8> for Color {
    type Error = ();

    fn try_from(v: u8) -> Result<Self, Self::Error> {
        unsafe { std::mem::transmute(v) }
    }
}

impl TryFrom<u8> for PieceType {
    type Error = ();

    fn try_from(v: u8) -> Result<Self, Self::Error> {
        unsafe { std::mem::transmute(v) }
    }
}

impl Pieces {
    pub fn color(&self) -> Color {
        (*self as u8 % 2).try_into().unwrap()
    }
    pub fn piecetype(&self) -> PieceType {
        (*self as u8 / 2).try_into().unwrap()
    }
    pub fn from_viri(id: u8) -> Pieces {
        ((id >> 3) | (id & 0b111) << 1).try_into().unwrap()
    }
    pub fn from_pc_color(pc: PieceType, c: Color) -> Pieces {
        ((pc as u8) << 1 | c as u8).try_into().unwrap()
    }
}

impl TryFrom<u8> for Pieces {
    type Error = ();

    fn try_from(v: u8) -> Result<Self, Self::Error> {
        unsafe { std::mem::transmute(v) }
    }
}

#[derive(Clone, Copy)]
pub struct Position{
    pub bbs: [u64; 8],
    pub mailbox: [u8; 64],
    pub score: i16,
    pub bound: u8,
    pub rule50: u8,
    pub bm: u16,
    pub stm: bool,
    pub depth: u8,
    pub ep: u8,
    pub castle_mask: u64,
}

impl Default for Position {
    fn default() -> Self {
        Self {
            bbs: [0; 8],
            mailbox: [Pieces::Void as u8; 64],
            score: 0,
            bound: 0,
            rule50: 0,
            bm: 0,
            stm: true,
            depth: 0,
            ep: 64,
            castle_mask: 0,
        }
    }
}

const KINGPOS_CASTLE: [u8; 2] = [2, 6];
const ROOKPOS_CASTLE: [u8; 2] = [3, 5];

impl Position {
    pub fn piece(&self, idx: usize) -> Pieces {
        self.mailbox[idx].try_into().unwrap()
    }
    pub fn push(&mut self, idx: u32, piece: Pieces) {
        assert!(self.mailbox[idx as usize] == Pieces::Void as u8);
        self.mailbox[idx as usize] = piece as u8;
        self.bbs[piece.color() as usize] |= 1 << idx;
        self.bbs[piece.piecetype() as usize + 2] |= 1 << idx;
    }

    pub fn update_ep(&mut self, newep: usize) {
        self.ep = newep as u8;
    }
    pub fn update_stm(&mut self, newstm: usize) {
        self.stm = newstm != 0;
    }
    pub fn turn_stm(&mut self) {
        self.stm = !self.stm;
    }
    pub fn update_castle_rights(&mut self, castle_idx: usize) {
        self.castle_mask |= 1 << castle_idx as u64;
    }

    fn erase(&mut self, idx: usize){
        let erased: Pieces = self.piece(idx);
        if erased != Pieces::Void {
            self.bbs[erased.piecetype() as usize + 2] ^= 1 << idx as u64;
            self.bbs[erased.color() as usize] ^= 1 << idx as u64;
        }
        self.mailbox[idx] = Pieces::Void as u8;
    }

    fn erasesure(&mut self, idx: usize){
        let erased: Pieces = self.piece(idx);
        self.bbs[erased.piecetype() as usize + 2] ^= 1 << idx as u64;
        self.bbs[erased.color() as usize] ^= 1 << idx as u64;
        self.mailbox[idx] = Pieces::Void as u8;
    }

    pub fn play(&mut self, mv: u16) {
        let from = (mv & 0x3f) as u8;
        let to = ((mv >> 6) & 0x3f) as u8;
        let promo = (mv >> 12) & 0b11;
        let flag = mv >> 14;
        let piece: Pieces = self.piece(from as usize);
        assert!(piece != Pieces::Void);
        let capture: Pieces = self.piece(to as usize);
        let _reset: bool = (capture != Pieces::Void && flag != 2) || piece.piecetype() == PieceType::Pawn;
        // println!("from = {} to = {} piece = {} (piecetype = {}) type = {} promo = {} ep = {}", from, to, piece as u8, piece.piecetype() as u8, flag, promo, self.ep);
        // println!("=> poss = {} & {}", piece.piecetype() == 0, ((to^from)&0b10000) != 0);

        if flag == 0 {

            let masktofrom: u64 = 
                (1 << to as u64) | (1 << from as u64) | 
                (piece.piecetype() == PieceType::King) as u64 * (0xff << piece.color() as u64 * 7);
            
            let mask: u64 = self.castle_mask & masktofrom;
            self.castle_mask ^= mask;

            self.ep = (piece.piecetype() == PieceType::Pawn && ((to ^ from) & 0b10_000) != 0) as u8 * to as u8;
            self.ep += 64*(self.ep == 0) as u8;

            self.erasesure(from as usize);
            self.erase(to as usize);
            self.push(to as u32, piece);
        } else if flag == 1 {
            assert!(self.ep < 64);
            assert!(self.mailbox[self.ep as usize]/2 == 0);
            self.erasesure(from as usize);
            self.erasesure(self.ep as usize);
            self.push(to as u32, piece);
            self.ep = 64;
        } else if flag == 2 {

            let to_king = (from&56) | KINGPOS_CASTLE[(from < to) as usize];
            let to_rook = (from&56) | ROOKPOS_CASTLE[(from < to) as usize];

            let mask: u64 = (0xff << ((from & 56) >> 3) as u64) & self.castle_mask;
            self.castle_mask ^= mask;

            self.erasesure(from as usize);
            self.erasesure(to as usize);
            self.push(to_king as u32, piece);
            self.push(to_rook as u32, Pieces::from_pc_color(PieceType::Rook, piece.color()));

            self.ep = 64;
        } else {
            assert!(flag == 3);
            let to_piece = Pieces::from_pc_color(((promo+1) as u8).try_into().unwrap(), piece.color());
            if capture.piecetype() == PieceType::Rook && (self.castle_mask & (1 << to as u64) != 0){
                self.castle_mask &= !(1 << to as u64);
            }
            self.ep = 64;

            self.erasesure(from as usize);
            self.erase(to as usize);
            self.push(to as u32, to_piece);
        }

        self.turn_stm();
        // if !self.verify(){
        //     println!("from = {} to = {} piece = {} (piecetype = {}) type = {} promo = {} capture = {}", from, to, piece as u8, piece.piecetype() as u8, flag, promo, capture as u8);
        //     unreachable!();
        // }
    }
    pub fn tobullet(&self) -> ChessBoard {
        let mut res = ChessBoard::from_raw(self.bbs, self.stm as usize, self.score, self.bound as f32).unwrap();
        res.result = self.bound;
        res.score = self.score;
        res
    }
}