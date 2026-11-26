
WINNING_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6)
]


def check_winner(board):
    for a, b, c in WINNING_LINES:
        if board[a] == board[b] == board[c] and board[a] != "":
            return board[a]

    if "" not in board:
        return "draw"

    return None


def minimax(board, depth, is_maximizing, alpha, beta):
    winner = check_winner(board)

    if winner == "X":
        return 10 - depth

    if winner == "O":
        return depth - 10

    if winner == "draw":
        return 0

    if is_maximizing:
        best_score = float("-inf")

        for i in range(9):
            if board[i] == "":
                board[i] = "X"
                score = minimax(board, depth + 1, False, alpha, beta)
                board[i] = ""

                best_score = max(best_score, score)
                alpha = max(alpha, best_score)

                if beta <= alpha:
                    break

        return best_score

    else:
        best_score = float("inf")

        for i in range(9):
            if board[i] == "":
                board[i] = "O"
                score = minimax(board, depth + 1, True, alpha, beta)
                board[i] = ""

                best_score = min(best_score, score)
                beta = min(beta, best_score)

                if beta <= alpha:
                    break

        return best_score


def get_best_move(board):
    best_score = float("-inf")
    best_move = None

    for i in range(9):
        if board[i] == "":
            board[i] = "X"

            score = minimax(
                board, 0, False,
                float("-inf"), float("inf")
            )

            board[i] = ""

            if score > best_score:
                best_score = score
                best_move = i

    return best_move
