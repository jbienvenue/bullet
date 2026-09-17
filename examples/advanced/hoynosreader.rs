// based on the viri loader in bullet
use bullet_trainer::reader::DataReader;
use rand::seq::SliceRandom;
use std::fs::File;
use std::io::BufReader;
use std::sync::mpsc::{self, SyncSender};
use std::io::Read;

use bullet_lib::game::formats::bulletformat::ChessBoard;

#[derive(Clone)]
pub struct HoynosReader {
    file_paths: Vec<String>,
    buffer_size: usize,
    threads: usize,
}

impl HoynosReader {
    pub fn new(file_paths: Vec<String>, buffer_size_mb: usize, threads: usize) -> Self {
        Self {
            file_paths,
            buffer_size: buffer_size_mb * 1024 * 1024 / size_of::<ChessBoard>() / 2,
            threads,
        }
    }
}
const POSITION_RECORD_SIZE: usize = 28;
pub fn read_game_bytes(reader: &mut impl Read, buf: &mut Vec<u8>) -> std::io::Result<bool> {
    buf.resize(POSITION_RECORD_SIZE, 0);

    if reader.read(&mut buf[..1])? == 0 {
        return Ok(false);
    }
    reader.read_exact(&mut buf[1..POSITION_RECORD_SIZE])?;

    /*let count = u16::from_le_bytes([buf[COUNT_BYTES], buf[COUNT_BYTES + 1]]) as usize;

    buf.resize(POSITION_RECORD_SIZE + 4 * count, 0);
    reader.read_exact(&mut buf[POSITION_RECORD_SIZE..])?;*/

    Ok(true)
}

impl DataReader<ChessBoard> for HoynosReader {
    fn read_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, _: usize, mut f: F) {
        let mut shuffle_buffer = Vec::new();
        shuffle_buffer.reserve_exact(self.buffer_size);

        let file_paths = self.file_paths.clone();
        let buffer_size = self.buffer_size;
        let threads = self.threads;

        let (sender, receiver) = mpsc::sync_channel::<Vec<Vec<u8>>>(4);
        let (msg_sender, msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            let mut games = Vec::new();

            'dataloading: loop {
                for file_path in &file_paths {
                    let mut reader = BufReader::new(File::open(file_path.as_str()).unwrap());

                    loop {
                        let mut buf = Vec::new();
                        if !read_game_bytes(&mut reader, &mut buf).unwrap_or(false) {
                            break;
                        }

                        games.push(buf);

                        if games.len().is_multiple_of(8192 * threads) {
                            if msg_receiver.try_recv().unwrap_or(false) || sender.send(games).is_err() {
                                break 'dataloading;
                            }

                            games = Vec::new();
                        }
                    }
                }
            }
        });

        let (game_sender, game_receiver) = mpsc::sync_channel::<Vec<ChessBoard>>(4 * self.threads);
        let (game_msg_sender, game_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            'dataloading: while let Ok(games) = receiver.recv() {
                if game_msg_receiver.try_recv().unwrap_or(false) {
                    msg_sender.send(true).unwrap();
                    break 'dataloading;
                }

                convert_buffer(threads, &game_sender, &games);
            }
        });

        let (buffer_sender, buffer_receiver) = mpsc::sync_channel::<Vec<ChessBoard>>(0);
        let (buffer_msg_sender, buffer_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            'dataloading: while let Ok(game) = game_receiver.recv() {
                if buffer_msg_receiver.try_recv().unwrap_or(false) {
                    game_msg_sender.send(true).unwrap();
                    break 'dataloading;
                }

                if shuffle_buffer.len() + game.len() < shuffle_buffer.capacity() {
                    shuffle_buffer.extend_from_slice(&game);
                } else {
                    let diff = shuffle_buffer.capacity() - shuffle_buffer.len();
                    if diff > 0 {
                        shuffle_buffer.extend_from_slice(&game[..diff]);
                    }

                    shuffle_buffer.shuffle(&mut rand::rng());

                    if buffer_msg_receiver.try_recv().unwrap_or(false) || buffer_sender.send(shuffle_buffer).is_err() {
                        game_msg_sender.send(true).unwrap();
                        break 'dataloading;
                    }

                    shuffle_buffer = Vec::new();
                    shuffle_buffer.reserve_exact(buffer_size);
                    shuffle_buffer.extend_from_slice(&game[diff..]);
                }
            }
        });

        'dataloading: while let Ok(shuffle_buffer) = buffer_receiver.recv() {
            if f(&shuffle_buffer) {
                buffer_msg_sender.send(true).unwrap();
                break 'dataloading;
            }
        }

        drop(buffer_receiver);
    }
}

fn parse_positions(bytes: &[u8], out: &mut Vec<ChessBoard>) {
    assert_eq!(bytes.len(), 28);
    let occupancy: u64 = u64::from_le_bytes(bytes[..8].try_into().expect("wrong sized array"));
    let mut infos: u128 = u128::from_le_bytes(bytes[8..24].try_into().expect("wrong sized array"));
    let bm: u16 = u16::from_le_bytes(bytes[24..26].try_into().expect("wrong sized array"));
    let score: i16 = i16::from_le_bytes(bytes[26..28].try_into().expect("wrong sized array"));

    let depth: i32 = (infos%32) as i32;
    infos /= 32;

    let bound: u8 = (infos%3) as u8;
    infos /= 3;

    let count50: i32 = (infos%100) as i32;
    infos /= 100;

    let stm: usize = (infos%2) as usize;
    infos /= 2;

    let mut kingpos2: u8 = (infos % 31) as u8;
    infos /= 31;

    let kingpos1: u8 = (infos % 32) as u8;
    kingpos2 += (kingpos1 <= kingpos2) as u8;
    infos /= 32;

    let mut bbs: [u64; 8] = [0; 8];
    let mut mask: u64 = occupancy;
    let mut idx: usize = 0;
    while mask > 0 {
        let sq = mask.trailing_zeros();
        let sqmask: u64 = 1 << sq as u64;
        let mut piece: u8;
        if idx == kingpos1 as usize {
            piece = 5*2;
        }else if idx == kingpos2 as usize {
            piece = 5*2+1;
        }else {
            piece = (infos % 13) as u8;
            if piece == 11 {
                if sq / 8 == 0 || sq / 8 == 7 {
                    piece = 3; // rook
                }else {
                    piece = 0; // pawn
                }
                piece = piece * 2 + (sq / 8 >= 4) as u8;
            }
            infos /= 13;
        }
        bbs[(piece%2) as usize] |= sqmask;
        bbs[(piece >> 1) as usize + 2] |= sqmask;

        idx += 1;
        mask &= mask-1;
    }
    let mut board: ChessBoard = ChessBoard::from_raw(
        bbs,
        stm,
        score,
        bound as f32 / 2.0,
    ).unwrap();
    board.result = bound;
    out.push(board);
}

fn convert_buffer(threads: usize, sender: &SyncSender<Vec<ChessBoard>>, games: &[Vec<u8>]) {
    let chunk_size = games.len().div_ceil(threads);

    std::thread::scope(|s| {
        for chunk in games.chunks(chunk_size) {
            let this_sender = sender.clone();
            s.spawn(move || {
                let mut buffer = Vec::new();

                for game_bytes in chunk {
                    parse_positions(game_bytes, &mut buffer);
                }

                this_sender.send(buffer)
            });
        }
    });
}