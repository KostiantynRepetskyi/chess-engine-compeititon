import board_tools as bt
import numpy as np
from numba import njit, int64, int32, uint32, uint64


_DEBRUIJN64 = np.uint64(0x03F79D71B4CB0A89)
_INDEX64 = np.array([
    0, 1, 48, 2, 57, 49, 28, 3,
    61, 58, 50, 42, 38, 29, 17, 4,
    62, 55, 59, 36, 53, 51, 43, 22,
    45, 39, 33, 30, 24, 18, 12, 5,
    63, 47, 56, 27, 60, 41, 37, 16,
    54, 35, 52, 21, 44, 32, 23, 11,
    46, 26, 40, 15, 34, 20, 31, 10,
    25, 14, 19, 9, 13, 8, 7, 6,
], dtype=np.int32)


@njit(int32(uint64), cache=True)
def bit_scan_forward(lsb):
    """Index (0..63) of a one-hot uint64 value."""
    return _INDEX64[np.int32((lsb * _DEBRUIJN64) >> np.uint64(58))]








CENTER_CONTROL = np.array([
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0,5,5,5,5,0,0,
    0,0,5,10,10,5,0,0,
    0,0,5,10,10,5,0,0,
    0,0,5,5,5,5,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
], dtype=np.int32)



KING_CENTER_CONTROL = np.array([
    0,0,0,0,0,0,0,0,
    0,5,5,5,5,5,5,0,
    0,5,10,10,10,10,5,0,
    0,5,10,15,15,10,5,0,
    0,5,10,15,15,10,5,0,
    0,5,10,10,10,10,5,0,
    0,5,5,5,5,5,5,0,
    0,0,0,0,0,0,0,0,
], dtype=np.int32)


"""
Piece Value Constants
"""
# White Piece Values
PAWN_WHITE   = 100
KNIGHT_WHITE = 300
BISHOP_WHITE = 340
ROOK_WHITE   = 510
QUEEN_WHITE  = 875
KING_WHITE   = 20000

# Black Piece Values (Negative)
PAWN_BLACK   = -100
KNIGHT_BLACK = -300
BISHOP_BLACK = -340
ROOK_BLACK   = -510
QUEEN_BLACK  = -875
KING_BLACK   = -20000

"""
Turn Constants
"""
WHITE_TO_MOVE = 0
BLACK_TO_MOVE = 1





@njit(cache=True)
def bishop_visibility(sq, occ_all, occ_opp):
    """
    Count squares a bishop can 'see' from sq along diagonals.

    Preserves the previous inline behavior:
    - Empty squares count.
    - If the first blocker is an enemy piece, that square also counts (+1) and the ray stops.
    - If the first blocker is a friendly piece, it does not count and the ray stops.
    """
    f0 = sq & 7
    r0 = sq >> 3
    visible = 0

    for df, dr in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        for k in range(1, 8):
            ff = f0 + df * k
            rr = r0 + dr * k
            if ff > 7 or ff < 0 or rr > 7 or rr < 0:
                break
            target_sq = rr * 8 + ff
            ray_mask = np.uint64(1) << np.uint64(target_sq)
            if occ_all & ray_mask:
                if occ_opp & ray_mask:
                    visible += 1
                break
            visible += 1

    return visible


@njit(cache=True)
def eval_white_pawn(sq, white_pawn_count):
    endgame_delta = 20
    r = sq >> 3
    f = (sq & 7) + 1
    white_pawn_count[f] += 1

    center = CENTER_CONTROL[sq]
    white_delta = center + PAWN_WHITE

    endgame_delta -= center
    endgame_delta += r * 2
    return white_delta, endgame_delta


@njit(cache=True)
def eval_white_knight(sq):
    if sq < 8:
        return KNIGHT_WHITE - 10
    return KNIGHT_WHITE + CENTER_CONTROL[sq]


@njit(cache=True)
def eval_white_bishop(sq, occ_all, occ_black):
    white_delta = BISHOP_WHITE

    if sq < 8:
        white_delta -= 10

    vis = bishop_visibility(sq, occ_all, occ_black)
    if vis > 6:
        white_delta += 10
    elif sq > 31 and vis < 3:
        white_delta -= 100

    return white_delta


@njit(cache=True)
def eval_white_rook(sq, board_pieces):
    white_delta = ROOK_WHITE
    endgame_delta = 0

    if 48 <= sq <= 55:
        white_delta += 15

    if sq < 8:
        black_pawn_block = False
        white_pawn_block = False

        for i in range(sq + 8, 64, 8):
            imask = np.uint64(1) << np.uint64(i)
            if np.uint64(board_pieces[6]) & imask:
                black_pawn_block = True
            elif np.uint64(board_pieces[0]) & imask:
                white_pawn_block = True

        if not white_pawn_block:
            white_delta += 10
            if not black_pawn_block:
                white_delta += 10
            else:
                endgame_delta += 10

    return white_delta, endgame_delta


@njit(cache=True)
def eval_white_queen():
    return QUEEN_WHITE


@njit(cache=True)
def eval_white_king(sq, board_pieces, mask7):
    white_delta = KING_WHITE
    endgame_delta = KING_CENTER_CONTROL[sq]

    if sq == 6 and (np.uint64(board_pieces[3]) & mask7) == 0:
        white_delta += 10
        endgame_delta -= 10
    elif sq != 4:
        white_delta -= 15
        endgame_delta += 15

    return white_delta, endgame_delta


@njit(cache=True)
def eval_black_pawn(sq, black_pawn_count):
    endgame_delta = -20
    r = sq >> 3
    f = (sq & 7) + 1
    black_pawn_count[f] += 1

    center = CENTER_CONTROL[sq]
    black_delta = (-center) + PAWN_BLACK

    endgame_delta += center
    endgame_delta -= (7 - r) * 2
    return black_delta, endgame_delta


@njit(cache=True)
def eval_black_knight(sq):
    if sq > 55:
        return KNIGHT_BLACK + 10
    return KNIGHT_BLACK - CENTER_CONTROL[sq]


@njit(cache=True)
def eval_black_bishop(sq, occ_all, occ_white):
    black_delta = BISHOP_BLACK

    if sq > 55:
        black_delta += 10

    vis = bishop_visibility(sq, occ_all, occ_white)
    if vis > 6:
        black_delta -= 10
    elif sq < 32 and vis < 3:
        black_delta += 100

    return black_delta


@njit(cache=True)
def eval_black_rook(sq, board_pieces):
    black_delta = ROOK_BLACK
    endgame_delta = 0

    if 8 <= sq <= 15:
        black_delta -= 15

    if sq > 55:
        black_pawn_block = False
        white_pawn_block = False

        for i in range(sq - 8, -1, -8):
            imask = np.uint64(1) << np.uint64(i)
            if np.uint64(board_pieces[6]) & imask:
                black_pawn_block = True
            elif np.uint64(board_pieces[0]) & imask:
                white_pawn_block = True

        if not black_pawn_block:
            black_delta -= 10
            if not white_pawn_block:
                black_delta -= 10
            else:
                endgame_delta -= 10

    return black_delta, endgame_delta


@njit(cache=True)
def eval_black_queen():
    return QUEEN_BLACK


@njit(cache=True)
def eval_black_king(sq, board_pieces, mask63):
    black_delta = KING_BLACK
    endgame_delta = -KING_CENTER_CONTROL[sq]

    if sq == 62 and (np.uint64(board_pieces[9]) & mask63) == 0:
        black_delta -= 10
        endgame_delta += 10
    elif sq != 50:
        black_delta += 15
        endgame_delta -= 15

    return black_delta, endgame_delta


@njit(int32(int64[:], int64[:], uint32))
def evaluation_function(board_pieces, board_occupancy, side_to_move):
    """
    Args:
        board_pieces: Array of 12 bitboards (piece locations) – Do not modify
        board_occupancy: Array of 3 bitboards (White, Black, All) – Do not modify
        side_to_move: 0 for White, 1 for Black
    
    Returns:
        int32: The score from the perspective of the side to move
               (Positive = Current player (side to move) is winning)
    """
    black_score = 0
    white_score = 0
    endgame_score_adjustment = 0

    white_pawn_count = np.zeros(10, dtype=np.int32)
    black_pawn_count = np.zeros(10, dtype=np.int32)

    white_queen = False
    black_queen = False





    occ_all = np.uint64(board_occupancy[2])
    occ_white = np.uint64(board_occupancy[0])
    occ_black = np.uint64(board_occupancy[1])
    mask7 = np.uint64(1) << np.uint64(7)
    mask63 = np.uint64(1) << np.uint64(63)

    for piece_type in range(12):
        piece_bitboard = np.uint64(board_pieces[piece_type])
        if piece_bitboard == 0:
            continue
        
        piece_idx = piece_type

        
        while piece_bitboard:
            lsb = piece_bitboard & (np.uint64(0) - piece_bitboard)
            sq = bit_scan_forward(lsb)
            piece_bitboard ^= lsb
           

            if piece_idx == 0:  # white pawn
                wd, ed = eval_white_pawn(sq, white_pawn_count)
                white_score += wd
                endgame_score_adjustment += ed







            elif piece_idx == 1:  # white knight
                white_score += eval_white_knight(sq)





            elif piece_idx == 2:  # white bishop
                white_score += eval_white_bishop(sq, occ_all, occ_black)







            elif piece_idx == 3:  # white rook
                wd, ed = eval_white_rook(sq, board_pieces)
                white_score += wd
                endgame_score_adjustment += ed








            elif piece_idx == 4:  # white queen
                white_queen = True
                white_score += eval_white_queen()







            elif piece_idx == 5:  # white king
                wd, ed = eval_white_king(sq, board_pieces, mask7)
                white_score += wd
                endgame_score_adjustment += ed








            elif piece_idx == 6:  # black pawn
                bd, ed = eval_black_pawn(sq, black_pawn_count)
                black_score += bd
                endgame_score_adjustment += ed









            elif piece_idx == 7:  # black knight
                black_score += eval_black_knight(sq)









            elif piece_idx == 8:  # black bishop
                black_score += eval_black_bishop(sq, occ_all, occ_white)









            elif piece_idx == 9:  # black rook
                bd, ed = eval_black_rook(sq, board_pieces)
                black_score += bd
                endgame_score_adjustment += ed









            elif piece_idx == 10:  # black queen
                black_queen = True
                black_score += eval_black_queen()









            else:  # piece_idx == 11: black king
                bd, ed = eval_black_king(sq, board_pieces, mask63)
                black_score += bd
                endgame_score_adjustment += ed







    score = white_score + black_score








    score_adjustment = 0

  

    for i in range(1, 9):
        wn = white_pawn_count[i]
        bn = black_pawn_count[i]

        if wn != 0:

            if wn >= 2:
                score_adjustment -= 10 * (wn - 1)
                endgame_score_adjustment -= 10 

            if black_pawn_count[i + 1] == 0 and black_pawn_count[i - 1] == 0 and black_pawn_count[i] == 0:
                score_adjustment += 10
                endgame_score_adjustment += 10


        if bn != 0:

            if bn >= 2:
                score_adjustment += 10 * (bn - 1)
                endgame_score_adjustment += 10 

            if white_pawn_count[i + 1] == 0 and white_pawn_count[i - 1] == 0 and white_pawn_count[i] == 0:
                score_adjustment -= 10
                endgame_score_adjustment -= 10

    score += score_adjustment







    if not white_queen and not black_queen and -1500 + KING_BLACK < black_score and white_score < 1500 + KING_WHITE:
        score += endgame_score_adjustment





    if side_to_move == BLACK_TO_MOVE:
        return -score
    return score
    



    

    
