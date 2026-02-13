
    







import board_tools as bt
import numpy as np
from numba import njit, int64, int32, uint32, uint64

# --- bitboard iteration helper (Numba-friendly) ---
# de Bruijn bit-scan for 64-bit: index of isolated least-significant 1-bit
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
                endgame_score_adjustment += 20
                r = sq >> 3
                f = (sq & 7) + 1
                white_pawn_count[f] += 1

                white_score += CENTER_CONTROL[sq]
                white_score += PAWN_WHITE

                endgame_score_adjustment -= CENTER_CONTROL[sq]
                endgame_score_adjustment += r * 2







            elif piece_idx == 1:  # white knight
                if sq < 8:
                    white_score -= 10
                else:
                    white_score += CENTER_CONTROL[sq]
                white_score += KNIGHT_WHITE





            elif piece_idx == 2:  # white bishop
                if sq < 8:
                    white_score -= 10

                f0 = sq & 7
                r0 = sq >> 3
                visible_empty_squares = 0

                for i, j in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    for k in range(1, 8):
                        ff = f0 + i * k
                        rr = r0 + j * k
                        if ff > 7 or ff < 0 or rr > 7 or rr < 0:
                            break
                        target_sq = rr * 8 + ff
                        ray_mask = np.uint64(1) << np.uint64(target_sq)
                        if occ_all & ray_mask:
                            if occ_black & ray_mask:
                                visible_empty_squares += 1
                            break
                        visible_empty_squares += 1

                if visible_empty_squares > 6:
                    white_score += 10
                elif sq > 31 and visible_empty_squares < 3:
                    white_score -= 100

                white_score += BISHOP_WHITE







            elif piece_idx == 3:  # white rook
                if 48 <= sq <= 55:
                    white_score += 15

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
                        white_score += 10
                        if not black_pawn_block:
                            white_score += 10
                        else:
                            endgame_score_adjustment += 10

                white_score += ROOK_WHITE








            elif piece_idx == 4:  # white queen
                white_queen = True
                white_score += QUEEN_WHITE







            elif piece_idx == 5:  # white king
                white_score += KING_WHITE
                endgame_score_adjustment += KING_CENTER_CONTROL[sq]

                if sq == 6 and (np.uint64(board_pieces[3]) & mask7) == 0:
                    white_score += 10
                    endgame_score_adjustment -= 10
                elif sq != 4:
                    white_score -= 15
                    endgame_score_adjustment += 15








            elif piece_idx == 6:  # black pawn
                endgame_score_adjustment -= 20
                r = sq >> 3
                f = (sq & 7) + 1
                black_pawn_count[f] += 1

                black_score -= CENTER_CONTROL[sq]
                black_score += PAWN_BLACK

                endgame_score_adjustment += CENTER_CONTROL[sq]
                endgame_score_adjustment -= (7 - r) * 2









            elif piece_idx == 7:  # black knight
                if sq > 55:
                    black_score += 10
                else:
                    black_score -= CENTER_CONTROL[sq]
                black_score += KNIGHT_BLACK









            elif piece_idx == 8:  # black bishop
                if sq > 55:
                    black_score += 10

                f0 = sq & 7
                r0 = sq >> 3
                visible_empty_squares = 0

                for i, j in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    for k in range(1, 8):
                        ff = f0 + i * k
                        rr = r0 + j * k
                        if ff > 7 or ff < 0 or rr > 7 or rr < 0:
                            break
                        target_sq = rr * 8 + ff
                        ray_mask = np.uint64(1) << np.uint64(target_sq)
                        if occ_all & ray_mask:
                            if occ_white & ray_mask:
                                visible_empty_squares += 1
                            break
                        visible_empty_squares += 1

                if visible_empty_squares > 6:
                    black_score -= 10
                elif sq < 32 and visible_empty_squares < 3:
                    black_score += 100

                black_score += BISHOP_BLACK









            elif piece_idx == 9:  # black rook
                if 8 <= sq <= 15:
                    black_score -= 15

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
                        black_score -= 10
                        if not white_pawn_block:
                            black_score -= 10
                        else:
                            endgame_score_adjustment -= 10

                black_score += ROOK_BLACK









            elif piece_idx == 10:  # black queen
                black_queen = True
                black_score += QUEEN_BLACK









            else:  # piece_idx == 11: black king
                endgame_score_adjustment -= KING_CENTER_CONTROL[sq]
                black_score += KING_BLACK

                if sq == 62 and (np.uint64(board_pieces[9]) & mask63) == 0:
                    black_score -= 10
                    endgame_score_adjustment += 10
                elif sq != 50:
                    black_score += 15
                    endgame_score_adjustment -= 15







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
    



    

    
