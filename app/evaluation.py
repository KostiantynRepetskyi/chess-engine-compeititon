
    







import board_tools as bt
import numpy as np
from numba import njit, int64, int32, uint32

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
    0,5,10,20,20,10,5,0,
    0,5,10,20,20,10,5,0,
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
BISHOP_WHITE = 360
ROOK_WHITE   = 510
QUEEN_WHITE  = 875
KING_WHITE   = 20000

# Black Piece Values (Negative)
PAWN_BLACK   = -100
KNIGHT_BLACK = -300
BISHOP_BLACK = -360
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

   

    for sq in range(64):
        piece_id = bt.get_piece(board_pieces, sq)
        if piece_id == 0:
            continue




        if piece_id == bt.WHITE_PAWN:
            r = sq // 8

            white_score += CENTER_CONTROL[sq]
            white_score += PAWN_WHITE

            endgame_score_adjustment -= CENTER_CONTROL[sq]
            endgame_score_adjustment += r * 2






        elif piece_id == bt.WHITE_KNIGHT:
            if sq < 8:
                white_score -= 10
            else:
                white_score += CENTER_CONTROL[sq]
            white_score += KNIGHT_WHITE







        elif piece_id == bt.WHITE_BISHOP:
            if sq < 8:
                white_score -= 10
            white_score += BISHOP_WHITE







        elif piece_id == bt.WHITE_ROOK:
            if 48 <= sq <= 55:
                white_score += 20

            if sq < 8:
                black_pawn_block = False
                white_pawn_block = False

                for i in range(sq + 8, 64, 8):
                    if bt.get_piece(board_pieces, i) == bt.BLACK_PAWN:
                        black_pawn_block = True
                    elif bt.get_piece(board_pieces, i) == bt.WHITE_PAWN:
                        white_pawn_block = True

                if not white_pawn_block:
                    white_score += 10
                    if not black_pawn_block:
                        white_score += 10
            white_score += ROOK_WHITE






        elif piece_id == bt.WHITE_QUEEN:
            white_score += QUEEN_WHITE







        elif piece_id == bt.WHITE_KING:
            white_score += KING_WHITE
            endgame_score_adjustment += KING_CENTER_CONTROL[sq]

            if sq == 6 and bt.get_piece(board_pieces, 7) != bt.WHITE_ROOK:
                white_score += 15
                endgame_score_adjustment -= 15

      




        elif piece_id == bt.BLACK_PAWN:
            r = sq // 8

            black_score -= CENTER_CONTROL[sq]
            black_score += PAWN_BLACK

            endgame_score_adjustment += CENTER_CONTROL[sq]
            endgame_score_adjustment -= (7 - r) * 2






        elif piece_id == bt.BLACK_KNIGHT:
            if sq > 55:
                black_score += 10
            else:
                black_score -= CENTER_CONTROL[sq]
            black_score += KNIGHT_BLACK






        elif piece_id == bt.BLACK_BISHOP:
            if sq > 55:
                black_score += 10
            black_score += BISHOP_BLACK




        elif piece_id == bt.BLACK_ROOK:
            if 8 <= sq <= 15:
                black_score -= 20

            if sq > 55:
                black_pawn_block = False
                white_pawn_block = False

                for i in range(sq - 8, -1, -8):
                    if bt.get_piece(board_pieces, i) == bt.BLACK_PAWN:
                        black_pawn_block = True
                    elif bt.get_piece(board_pieces, i) == bt.WHITE_PAWN:
                        white_pawn_block = True

                if not black_pawn_block:
                    black_score -= 10
                    if not white_pawn_block:
                        black_score -= 10         
            black_score += ROOK_BLACK






        elif piece_id == bt.BLACK_QUEEN:
            black_score += QUEEN_BLACK






        elif piece_id == bt.BLACK_KING:
            endgame_score_adjustment -= KING_CENTER_CONTROL[sq]
            black_score += KING_BLACK

            if sq == 62 and bt.get_piece(board_pieces, 63) != bt.BLACK_ROOK:
                black_score -= 15
                endgame_score_adjustment += 15






    score = white_score + black_score


    if -1000 + KING_BLACK < black_score and white_score < 1000 + KING_WHITE:
        score += endgame_score_adjustment

    

    if side_to_move == BLACK_TO_MOVE:
        return -score
    return score
    



    

    
