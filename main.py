import cv2
import pygame
import sys
from detection.board_detection import load_board_model, detect_board
from detection.piece_detection import load_piece_model, detect_pieces_and_get_positions
from board_display.board_display import draw_board, draw_pieces

# === Init models
board_model = load_board_model()
piece_model = load_piece_model()

# === Init Pygame
pygame.init()
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 880
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Bàn cờ Tướng")
font = pygame.font.SysFont("simsun", 28)
clock = pygame.time.Clock()

# === Init camera
cap = cv2.VideoCapture("http://192.168.1.97:4747/video")
if not cap.isOpened():
    print("❌ Không mở được webcam.")
    sys.exit()

pieces = []  # danh sách quân cờ sẽ update mỗi khung hình

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 5 == 0:
        board_img, has_board = detect_board(board_model, frame)
        if has_board:
            pieces = detect_pieces_and_get_positions(piece_model, board_img)

    # Pygame vẫn render mọi frame
    screen.fill((255, 255, 255))
    draw_board(screen)
    draw_pieces(screen, font, pieces)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            sys.exit()

    pygame.display.flip()
    clock.tick(30)


cap.release()
pygame.quit()
