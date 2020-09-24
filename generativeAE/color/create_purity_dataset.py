# encoding: utf-8
'''
Create a font dataset for trainding
Content / size / color(Font) / color(background) / style
E.g. A / 64/ red / blue / arial
potential : position (x, y) bold, rotation

'''
import os
import pygame

os.environ["SDL_VIDEODRIVER"] = "dummy"
'''reference'''
# color 10 (back ground and font)
'''
Colors = {'red': (220, 20, 60), 'orange': (255,165,0), 'Yellow': (255,255,0), 'green': (0,128,0), 'cyan' : (0,255,255),
         'blue': (0,0,255), 'purple': (128,0,128), 'pink': (255,192,203), 'chocolate': (210,105,30), 'silver': (192,192,192)}
         '''
Colors = {'red': (255, 0, 0), 'blue': (0, 0, 255), 'green': (0, 255, 0)}
# font_dir = '/home2/fonts_dataset_new'
font_dir = '/lab/tmpig23b/u/yao-data/generationAE/dataset/purity'
if not os.path.exists(font_dir):
    os.makedirs(font_dir)

img_size = 128

pygame.init()
screen = pygame.display.set_mode((img_size, img_size))  # image size Fix(128 * 128)

for back_color in Colors.keys():  # 4th round for back_color
    try:
        # 1 set back_color
        screen.fill(Colors[back_color])  # background color
        # 2 set letter

        # screen.blit(rtext, (img_size/2, img_size/2))
        # screen.blit(rtext, (img_size / 4, 0))
        # screen.blit(rtext, (10, 0)) # because
        # E.g. A / 64/ red / blue / arial
        img_name = back_color + ".png"
        img_path = os.path.join(font_dir, back_color)
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        pygame.image.save(screen, os.path.join(img_path, img_name))
    except:
        print(back_color)

# screen.fill((255,255,255)) # background color
# start, end = (97, 255) # 汉字编码范围
# for codepoint in range(int(start), int(end)):
#     word = chr(codepoint)
#     font = pygame.font.SysFont("arial", 64) # size and bold or not
#     # font = pygame.font.Font("msyh.ttc", 64)
#     rtext = font.render(word, True, (0, 0, 0), (255, 255, 255))
#     # pygame.image.save(rtext, os.path.join(chinese_dir, word + ".png"))
#     screen.blit(rtext, (300, 300))
#     pygame.image.save(screen, os.path.join(chinese_dir, word + ".png"))
