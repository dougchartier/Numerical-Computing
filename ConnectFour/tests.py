import connectfour

"""
    Tests for conenctfour.py.
    
"""

def board_check_regression_test(width=BOARD_WIDTH, height=BOARD_HEIGHT):
    board = make_board()

    board[board_xy_to_ind(5,2)] = 1
    board[board_xy_to_ind(4,2)] = 1
    board[board_xy_to_ind(3,2)] = 1
    board[board_xy_to_ind(2,2)] = 1

    print_board(board)
    cur_player = 1
    start_row = 4
    start_col = 2

    if has_player_won(start_row,start_col,cur_player,board):
        print('Test passed.')
    else:
        print('Test failed.')

def adj_syms_regression_test(width=BOARD_WIDTH, height=BOARD_HEIGHT):
    # Test boards
    boards = [
        [
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 2, 0, 2, 1, 0,
            0, 0, 1, 0, 1, 2, 0,
            0, 2, 1, 1, 2, 1, 0,
            0, 1, 2, 2, 2, 0, 0
        ],
        [
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 2, 0, 0, 2,
            0, 0, 0, 2, 1, 0, 1,
            0, 0, 2, 1, 1, 0, 2
        ],
        [
            0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0,
            2, 1, 2, 0, 0, 0, 0,
            1, 1, 1, 2, 2, 0, 0,
            2, 1, 1, 1, 2, 0, 0,
            2, 2, 2, 2, 2, 0, 0
        ],
        [
            0, 0, 0, 0, 1, 1, 0,
            0, 0, 0, 2, 1, 2, 0,
            0, 0, 2, 1, 1, 1, 2,
            0, 0, 1, 2, 2, 1, 2,
            0, 1, 2, 2, 2, 1, 1,
            2, 1, 1, 2, 2, 2, 2
        ],
        [
            0, 1, 0, 0, 1, 1, 0,
            0, 1, 0, 2, 2, 2, 0,
            0, 2, 2, 1, 1, 1, 2,
            0, 2, 1, 2, 2, 2, 2,
            2, 1, 2, 2, 1, 1, 1,
            1, 1, 1, 1, 2, 2, 2
        ]
    ]

    # Test data: List of [start_row,col_col,player), max # of adjacent symbols, player to search for]
    test_data_l = [ [(4,3),3,1], [(5,5),2,1], [(3,1),4,1], [[1,4],5,1], [(3,3),4,2]]
    for i, test_data in enumerate(test_data_l):
        print(f'*** Running test {i}.')
        row, col = test_data[0]
        correct_max = test_data[1]
        player = test_data[2]
        max = max_consecutive_pieces(row, col, player, boards[i], width, height)
        if max == correct_max:
            print(f'Test {i} passed.\n')
        else:
            print(f'Test {i} failed: Correct count is {correct_max}.  Test run return {max}.\n')

adj_syms_regression_test()