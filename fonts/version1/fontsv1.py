import os
import pygame

# Attributes' values

# Background color
# Back_colors = {'red': (220, 20, 60), 'orange': (255, 165, 0), 'Yellow': (255, 255, 0), 'green': (0, 128, 0),
#           'cyan': (0, 255, 255),
#           'blue': (0, 0, 255), 'purple': (128, 0, 128), 'pink': (255, 192, 203), 'chocolate': (210, 105, 30),
#           'silver': (192, 192, 192)}
Back_colors = {'red': (220, 20, 60), 'Yellow': (255, 255, 0), 'green': (0, 128, 0),
          'cyan': (0, 255, 255),
          'blue': (0, 0, 255), 'pink': (255, 192, 203)
          }
# Foreground color
# Fore_colors = {'red': (255, 106, 106), 'orange': (255, 140, 0), 'Yellow': (205, 205, 0), 'green': (34, 139, 34),
#           'cyan': (0, 139, 139),
#           'blue': (0, 191, 255), 'purple': (216, 191, 216), 'pink': (255, 20, 147), 'chocolate': (165, 42, 42),
#           'silver': (192, 192, 192)}
Fore_colors = {'red1': (255, 106, 106), 'Yellow1': (205, 205, 0), 'green1': (34, 139, 34),
          'cyan1': (0, 139, 139),
          'blue1': (0, 191, 255), 'pink1': (255, 20, 147),
          }
# Sizes
Sizes = {'small': 40, 'medium': 50, 'large': 60}#456
# Fonts styles
    # style nearly over 100
All_fonts = pygame.font.get_fonts()
useless_fonts = ['notocoloremoji', 'droidsansfallback', 'gubbi', 'kalapi', 'lklug',  'mrykacstqurn', 'ori1uni','pothana2000','vemana2000',
                'navilu', 'opensymbol', 'padmmaa', 'raghumalayalam', 'saab', 'samyakdevanagari', 'arplukaihk', 'arplukaitw', 'arplukaitwmbe',
                 'arpluminghk', 'arplumingtw', 'arplumingtwmbe','chandas','dejavusansmono', 'dejavuserif', 'gargi', 'freesans',
                 'jamrul', 'khmerossystem', 'liberationsans', 'liberationsansnarrow', 'notosanscjkhkblack', 'notosanscjkhkdemilight',
                 'notosanscjkhklight', 'notosanscjkhkmedium', 'notosanscjkjpblack', 'notosanscjkjpdemilight', 'notosanscjkjpmedium',
                 'notosanscjkjplight', 'notosanscjkkrdemilight', 'notosanscjkkrblack','notosanscjkkrmedium', 'notosanscjkkrlight','notosanscjksc',
                 'notosanscjkscblack', 'notosanscjkscdemilight', 'notosanscjksclight', 'notosanscjkscmedium', 'notosanscjkscthin', 'notosanscjktc',
                 'notosanscjktcblack', 'notosanscjktcdemilight', 'notosanscjktclight', 'notosanscjktcmedium', 'notosanscjktcthin',
                 'notoserifcjksc', 'notoserifcjktc', 'padmaa', 'rekha', 'tlwgtypewriter', 'tlwgtypo', 'ubuntucondensed',
                 'ubuntumono','aakar','anjalioldlipi','chandas', 'chilanka', 'dyuthi','keraleeyam','khmeros','khmerossystem',
                 'kinnari','manjari','meera','nakula', 'rachana','suruma', 'tibetanmachineuni','uroob']
useless_fontsets = ['kacst', 'lohit', 'sam']

    # throw away the useless
    # for useless_font in useless_fonts:
    #     All_fonts.remove(useless_font)
temp = All_fonts.copy()
for useless_font in temp: # check every one
    for set in useless_fontsets:
        if set in useless_font:
            try:
                All_fonts.remove(useless_font)
            except:
                print(useless_font)
# Words lower case and upper case
Letters = list(range(65, 91)) + list(range(97, 123))
    # words 2
wd2 = 'ah am an as at be by do em en er fe go ha he hi id if in is it ko la ma me mu my na no of oh ok on oo op or pa pi qi re so to uh um un up ur us vu we wo xi ye yo zo ' \
      'AH AM AN AS AT BE BY DO EM EN ER FE GO HA HE HI ID IF IN IS IT KO LA MA ME MU MY NA NO OF OH OK ON OO OP OR PA PI QI RE SO TO UH UM UN UP UR US VU WE WO XI YE YO ZO'
Words2 = wd2.split(' ')
    # words 3
wd3 = 'the, and, for, are, but, not, you, all, any, can, had, her, was, one, our, out, day, get, has, him, his, how, man, new, now, old, see, two, way, who, boy, did, its, let, put, say, she, too, use, ' \
      'THE, AND, FOR, ARE, BUT, NOT, YOU, ALL, ANY, CAN, HAD, HER, WAS, ONE, OUR, OUT, DAY, GET, HAS, HIM, HIS, HOW, MAN, NEW, NOW, OLD, SEE, TWO, WAY, WHO, BOY, DID, ITS, LET, PUT, SAY, SHE, TOO, USE'
Words3 = wd3.split(', ')
    # words 1
Words1 = []
for i in Letters:
    Words1.append(chr(i))
Words = Words3 + Words2 + Words1
# location
# Locations = [(0, 0),(0, 1),(0, 2),
#              (1, 0),(1, 1),(1, 2),
#              (2, 0),(2, 1),(2, 2)]#only y axis
Locations = {'up':(1, 0),'mid':(1, 1),'down':(1, 2)}
# italic
Italics = [True, False]
# bold
Bold = [0, 1]
# texture
Textures = {'l': './l.png', 'r':'./r.png', 'u':'./u.png','None':None,}
# imgsize
img_size = 128
# save location
font_dir = '/home2/fonts_dataset_version1'
if not os.path.exists(font_dir):
    os.makedirs(font_dir)

pygame.init()
screen = pygame.display.set_mode((img_size, img_size))
cnt = 0
for word in Words: # 1st round for words
    print(word)
    for size in Sizes.keys():  # 2nd round for size
        print(size)
        for font_color in Fore_colors.keys():  # 3rd round for font_color
            for back_color in Back_colors.keys():  # 4th round for back_color
                # if not back_color == font_color:''' should not be same '''
                for font in All_fonts:  # 5th round for fonts
                    for italic_flag in Italics:
                        for location in Locations.keys():
                            for tx in Textures.keys():
                                if not font_color == back_color:
                                    cnt+=1
                                    print(cnt,'/136857600')
                                    try:
                                        # 1 set back_color
                                        screen.fill(Back_colors[back_color])  # background color
                                        if Textures[tx] is not None:
                                            texture = pygame.image.load(Textures[tx])
                                            screen.blit(texture, (-10, -10), special_flags=pygame.BLEND_RGB_MULT)
                                        # 2 set letter/word
                                        selected_letter = word
                                        # 3,4 set font and size
                                        selected_font = pygame.font.SysFont(font, Sizes[size], italic=italic_flag, bold=True) # size and bold or not
                                        font_size = selected_font.size(selected_letter);
                                        # 5 set font_color
                                        rtext = selected_font.render(selected_letter, True, Fore_colors[font_color])
                                        # rtext.set_alpha(0)
                                        # alphaimg = pygame.Surface(rtext.get_size(), pygame.SRCALPHA)
                                        # alphaimg.fill((255, 255, 255, 0))
                                        # rtext.blit(alphaimg,(0,0), special_flags=pygame.BLEND_RGB_MULT)


                                        # rtext.blit(texture,(0,0), special_flags=pygame.BLEND_RGB_MULT)
                                        # 6 render
                                        drawX = img_size / 2 - (font_size[0] / 2.0) + (Locations[location][0]-1) * 0.8 * (img_size / 2 - (font_size[0] / 2.0))
                                        drawY = img_size / 2 - (font_size[1] / 2.0) + (Locations[location][1]-1) * 0.8 * (img_size / 2 - (font_size[1] / 2.0))
                                        screen.blit(rtext, (drawX, drawY)) # because
                                        # E.g. A / 64/ red / blue / arial
                                        img_name = selected_letter + '_' + size + '_' + font_color + '_' + back_color + '_' \
                                                   + font + '_' + str(italic_flag) + '_' + str(location) + '_' + tx + ".png"
                                        img_path = os.path.join(font_dir, selected_letter, size, font_color, back_color, font, str(italic_flag), location, tx)
                                        if not os.path.exists(img_path):
                                            os.makedirs(img_path)
                                        pygame.image.save(screen, os.path.join(img_path, img_name))
                                    except:
                                        # print(letter, size, font_color, back_color, font)
                                        print(word, size, font_color, back_color, font)
                                else:
                                    break