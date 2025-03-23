import numpy as np
import colorama as cl

layouts = {"o":[[(0,1),(0,2),(1,1),(1,2)],[(0,1),(0,2),(1,1),(1,2)],
                [(0,1),(0,2),(1,1),(1,2)],[(0,1),(0,2),(1,1),(1,2)]],
           "i":[[(1,0),(1,1),(1,2),(1,3)],[(0,2),(1,2),(2,2),(3,2)],
                [(2,0),(2,1),(2,2),(2,3)],[(0,1),(1,1),(2,1),(3,1)]],
           "t":[[(0,1),(1,0),(1,1),(1,2)],[(0,1),(1,1),(1,2),(2,1)],
                [(1,0),(1,1),(1,2),(2,1)],[(0,1),(1,0),(1,1),(2,1)]],
           "l":[[(0,2),(1,0),(1,1),(1,2)],[(0,1),(1,1),(2,1),(2,2)],
                [(1,0),(1,1),(1,2),(2,0)],[(0,0),(0,1),(1,1),(2,1)]],
           "j":[[(0,0),(1,0),(1,1),(1,2)],[(0,1),(0,2),(1,1),(2,1)],
                [(1,0),(1,1),(1,2),(2,2)],[(0,1),(1,1),(2,0),(2,1)]],
           "s":[[(0,1),(0,2),(1,0),(1,1)],[(0,1),(1,1),(1,2),(2,2)],
                [(1,1),(1,2),(2,0),(2,1)],[(0,0),(1,0),(1,1),(2,1)]],
           "z":[[(0,0),(0,1),(1,1),(1,2)],[(0,2),(1,1),(1,2),(2,1)],
                [(1,0),(1,1),(2,1),(2,2)],[(0,1),(1,0),(1,1),(2,0)]]}

def align_to(pivot, layout):
    result = []
    for dot in layout:
        result.append((dot[0]+pivot[0],dot[1]+pivot[1]))
    return result

def dealign(piece, rotation):
    for y in range(-3,max([dot[0] for dot in piece[:-1]])):
        for x in range(-3,max([dot[1] for dot in piece[:-1]])):
            if layouts[piece[-1]][rotation] == [(dot[0]-y, dot[1]-x) for dot in piece[:-1]]:
                return (y,x)
            else:
                print(str((y,x)) + " not")

class Game:
    map = np.zeros((23,10))
    hold_type = ""
    current_piece = [(),(),(),(), ""]
    score = 0
    rot = 0
    shift = (0,0)

    prev_comb = ""
    b2b = 0
    b2b_comb = ""

    bag = []
    bag_index = 0

    def print_map(self):
        checkboard = True
        row_n = 0
        print("---"*11)
        for row in self.map:
            row_n += 1
            print((cl.Back.RED if row_n == 3 else cl.Fore.RESET) + "|", end="")
            for dot in row:
                print((cl.Back.YELLOW if dot == 1 else
                       cl.Back.CYAN if dot == 2 else
                       cl.Back.MAGENTA if dot == 3 else
                       cl.Back.YELLOW if dot == 4 else
                       cl.Back.BLUE if dot == 5 else
                       cl.Back.GREEN if dot == 6 else
                       cl.Back.RED if dot == 7 else
                       cl.Back.BLACK if checkboard else cl.Back.RESET) + "   ", end="")
                checkboard = not checkboard
            print((cl.Back.RED if row_n == 3 else cl.Fore.RESET)+"|" + cl.Back.RESET)
            checkboard = not checkboard
        print("Hold: " + self.hold_type)
        print("Score: " + str(self.score))
        print("Bag: " + str(self.bag) + f" Next {self.bag[self.bag_index]}")

    def place_dots(self, piece):
        for dot in piece[:-1]:
            self.map[dot] = 1 if piece[-1] == "o" else 2 if piece[-1] == "i"\
                else 3 if piece[-1] == "t" else 4 if piece[-1] == "l"\
                else 5 if piece[-1] == "j" else 6 if piece[-1] == "s"\
                else 7 if piece[-1] == "z" else 0

    def visible_current(self, boolean):
        if not boolean:
            for dot in self.current_piece[:-1]:
                self.map[dot] = 0
        else:
            self.place_dots(self.current_piece)

    def create_piece(self):
        if not self.bag:
            self.generate_7bag()
        ptype = self.bag[self.bag_index]
        p = layouts[ptype][0]
        ap = align_to((0,3),p)
        for dot in ap:
            if self.map[dot] != 0:
                print("< LOSS >")
                return False
        self.current_piece = ap + [ptype]
        self.visible_current(True)
        self.rot = 0
        self.bag_index += 1
        if self.bag_index == 7:
            self.generate_7bag()
            self.bag_index = 0
        return True

    def move_left(self):
        next = []
        self.visible_current(False)
        for y,x in self.current_piece[:-1]:
            next.append((y,x-1))
        for dot in next:
            if dot[1] < 0 or self.map[dot] != 0:
                next = self.current_piece[:-1]
                break
        next.append(self.current_piece[-1])
        self.current_piece = next
        self.visible_current(True)

    def move_right(self):
        next = []
        self.visible_current(False)
        for y,x in self.current_piece[:-1]:
            next.append((y,x+1))
        for dot in next:
            if dot[1] > 9 or self.map[dot] != 0:
                next = self.current_piece[:-1]
                break
        next.append(self.current_piece[-1])
        self.current_piece = next
        self.visible_current(True)

    def soft_drop(self):
        next = []
        self.visible_current(False)
        for y,x in self.current_piece[:-1]:
            next.append((y+1,x))
        for dot in next:
            if dot[0] > 22 or self.map[dot] != 0:
                next = self.current_piece[:-1]
                break
        next.append(self.current_piece[-1])
        self.current_piece = next
        self.visible_current(True)

    def rotate(self):
        self.visible_current(False)
        self.shift = dealign(self.current_piece, self.rot)
        for i in range(1,4):
            available = True
            rotation = self.rot + i
            while rotation > 3:
                rotation -= 4
            print(rotation)
            print(self.current_piece)
            print(self.shift)
            next = align_to(self.shift,layouts[self.current_piece[-1]][rotation]) + [self.current_piece[-1]]

            hor_alright = False
            ver_alright = False
            for dot in next[:-1]:
                if dot[1] < 0 and not hor_alright:
                    next = [(y,x-dot[1]) for y,x in next[:-1]] + [self.current_piece[-1]]
                    hor_alright = True
                if dot[1] > 9 and not hor_alright:
                    next = [(y,x-dot[1] + 9) for y,x in next[:-1]] + [self.current_piece[-1]]
                    hor_alright = True
                if dot[0] > 22 and not ver_alright:
                    next = [(y-dot[0],x) for y,x in next[:-1]] + [self.current_piece[-1]]
                    ver_alright = True
            for dot in next[:-1]:
                if self.map[dot]:
                    available = False
                    break
            if available:
                self.current_piece = next
                self.rot = rotation
                break
        self.visible_current(True)

    def drop(self):
        prev = []
        self.shift = dealign(self.current_piece, self.rot)
        while prev != self.current_piece:
            prev = self.current_piece
            self.soft_drop()
        self.clear_lines()
        self.create_piece()

    def clear_lines(self):
        self.shift = dealign(self.current_piece, self.rot)
        if self.current_piece[-1] == "t":
            corners_coords = align_to(self.shift,[(0,0),(0,2),(2,0),(2,2)])
            corners = [i for i in corners_coords if self.map[i] != 0 or i[1] < 0 or i[1] > 9]
            if len(corners) >= 3:
                self.prev_comb = "tspin"
                if len(corners) == 3:
                    if (corners == [i for i in align_to(self.shift,[(0,0),(2,0),(2,2)])
                        if self.map[i] != 0 or i[1] < 0 or i[1] > 9]
                        and (self.rot == 0 or self.rot == 1)) \
                            or (corners == [i for i in align_to(self.shift,[(0,2),(2,0),(2,2)])
                                if self.map[i] != 0 or i[1] < 0 or i[1] > 9]
                                and (self.rot == 0 or self.rot == 3)) \
                            or (corners == [i for i in align_to(self.shift,[(0,0),(0,2),(2,2)])
                                if self.map[i] != 0 or i[1] < 0 or i[1] > 9]
                                and (self.rot == 3 or self.rot == 2)) \
                            or (corners == [i for i in align_to(self.shift,[(0,0),(0,2),(2,0)])
                                if self.map[i] != 0 or i[1] < 0 or i[1] > 9]
                                and (self.rot == 1 or self.rot == 2)):
                        self.prev_comb = "minitspin"
            else:
                self.prev_comb = ""
        else:
            self.prev_comb = ""
        cleared = 0
        row_n = 22
        while row_n > -1:
            if len([dot for dot in self.map[row_n] if dot]) == 10:
                for uprow_n in range(row_n-1, -1, -1):
                    self.map[uprow_n+1] = [self.map[uprow_n][i] for i in range(10)]
                self.map[0] = [0 for _ in range(10)]
                cleared += 1
            else:
                row_n -= 1
        if cleared == 4:
            self.prev_comb = "tetris"

        if self.prev_comb == "" and cleared == 1:
            self.score += 100
        elif self.prev_comb == "minitspin" and cleared == 1:
            self.score += 200
        elif self.prev_comb == "" and cleared == 2:
            self.score += 300
        elif self.prev_comb == "minitspin" and cleared == 2 and self.b2b_comb != "tspin":
            self.score += 400
        elif self.prev_comb == "minitspin" and cleared == 2 and self.b2b_comb == "tspin":
            self.score += 600
        elif self.prev_comb == "" and cleared == 3:
            self.score += 500
        elif self.prev_comb == "tetris" and  self.b2b_comb != "tetris":
            self.score += 800
        elif self.prev_comb == "tspin" and cleared == 1 and self.b2b_comb != "tspin":
            self.score += 800
        elif self.prev_comb == "tspin" and cleared == 2 and self.b2b_comb != "tspin":
            self.score += 1200
        elif self.prev_comb == "tspin" and cleared == 1 and self.b2b_comb == "tspin":
            self.score += 1200
        elif self.prev_comb == "tetris" and self.b2b_comb == "tetris":
            self.score += 1200
        elif self.prev_comb == "tspin" and cleared == 3 and self.b2b_comb != "tspin":
            self.score += 1600
        elif self.prev_comb == "tspin" and cleared == 2 and self.b2b_comb == "tspin":
            self.score += 1800
        elif self.prev_comb == "tspin" and cleared == 3 and self.b2b_comb == "tspin":
            self.score += 2400

        if cleared:
            self.b2b_comb = "tspin" if self.prev_comb == "tspin" or self.prev_comb == "minitspin" else self.prev_comb

    def generate_7bag(self):
        self.bag = []
        standart = ["i","o","t","s","z","l","j"]
        for i in range(7):
            type = np.random.choice(standart)
            standart.remove(type)
            self.bag.append(type)

    def swap_hold(self):
        _bht =  self.hold_type
        self.visible_current(False)
        if _bht == "":
            self.hold_type = self.current_piece[-1]
            if not self.bag:
                self.generate_7bag()
            ptype = self.bag[self.bag_index]
            p = layouts[ptype][0]
            ap = align_to((0,3),p)
            self.current_piece = ap + [ptype]
            self.visible_current(True)
            self.rot = 0
            self.bag_index += 1
            if self.bag_index == 7:
                self.generate_7bag()
                self.bag_index = 0
        else:
            self.hold_type = self.current_piece[-1]
            self.current_piece = align_to((0,3),layouts[_bht][0]) + [_bht]
            self.rot = 0
            self.visible_current(True)


# g = Game()
# g.create_piece()
# while True:
#     g.print_map()
#     move = input()
#     if move == "left":
#         g.move_left()
#     if move == "right":
#         g.move_right()
#     if move == "soft":
#         g.soft_drop()
#     if move == "drop":
#         g.drop()
#     if move == "rot":
#         g.rotate()
#     if move == "swap":
#         g.swap_hold()
#     if move == "quit":
#         break