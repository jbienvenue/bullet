use crate::sydposition::{
    PieceType, Pieces, Position
};
use crate::sydfilter::SydFilter;
use std::io::{Cursor, Read};


use bullet_lib::game::formats::bulletformat::ChessBoard;

pub fn parse_tree(cursor: &mut Cursor<Vec<u8>>, out: &mut Vec<ChessBoard>, pos:&mut Position, filter: &SydFilter){
    let mut buff: [u8; 2] = [0; 2];

    cursor.read_exact(&mut buff).unwrap();
    let bm: u16 = u16::from_le_bytes(buff);

    cursor.read_exact(&mut buff).unwrap();
    let score: i16 = i16::from_le_bytes(buff);

    cursor.read_exact(&mut buff).unwrap();
    let info = buff[0];
    let nb_child = buff[1];

    pos.bm = bm;
    pos.bound = info%3;
    pos.depth = info/3;
    pos.score = score;

    if !filter.should_filter(pos) {
        out.push(pos.tobullet());
    }
    for _id_child in 0..nb_child {
        cursor.read_exact(&mut buff).unwrap();
        let mv: u16 = u16::from_le_bytes(buff);
        let mut newpos: Position = *pos;
        newpos.play(mv);
        parse_tree(cursor, out, &mut newpos, filter);
    }
}

pub fn parse_syd(reader: &Vec<u8>, res: &mut Vec<ChessBoard>, filter: &SydFilter) {

    let buff_4: [u8; 4] = reader[..4].try_into().unwrap();
    let size:u32 = u32::from_le_bytes(buff_4);
    let full = reader[4..].to_vec();
    assert!(full.len() == size as usize);

    let mut c = Cursor::new(full);
    let mut marlin: [u8; 32] = [0; 32];
    c.read_exact(&mut marlin).unwrap();
    
    let mut startpos: Position = Position::default();
    let occupancy = u64::from_le_bytes(marlin[..8].try_into().unwrap());
    let mut mask = occupancy;
    for id_piece in 0..occupancy.count_ones() {
        let pos = mask.trailing_zeros();
        let mut piece = Pieces::from_viri((marlin[(8+id_piece/2) as usize] >> 4 * (id_piece%2)) & 0b1111);
        //println!("piece={}", piece);
        if piece.piecetype() == PieceType::Castle {
            startpos.update_castle_rights(pos as usize);
            piece = Pieces::from_pc_color(PieceType::Rook, piece.color());
        }
        startpos.push(pos, piece);

        mask &= mask - 1;
    }
    let stmep = marlin[24];
    let mut epsquare = stmep & 0b1111111;
    let stm: bool = (stmep >> 7) != 0;
    if epsquare != 64 {
        if stm {
            epsquare += 8;
        } else {
            epsquare -= 8;
        }
    }
    startpos.update_ep(epsquare as usize);
    startpos.update_stm(stm as usize);
    startpos.rule50 = marlin[25];
    parse_tree(&mut c, res, &mut startpos, filter);

}