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
    lines_readiness = 0

    bag = []
    pocket = []

    pool_record = []

    lost = False

    collisions = 0
    moves_memory = [6 for _ in range(20)]

    def reset(self, replay):
        self.map = np.zeros((23,10))
        self.hold_type = ""
        self.current_piece = [(),(),(),(), ""]
        self.score = 0
        self.rot = 0
        self.shift = (0,0)

        self.prev_comb = ""
        self.b2b = 0
        self.b2b_comb = ""
        self.lines_readiness = 0

        self.generate_7bag()
        self.generate_pocket()

        self.lost = False

        if replay:
            if len(self.pool_record) >= 7:
                self.bag = self.pool_record[:7]
                self.pocket = self.pool_record[7:]
            self.pool_record = self.bag[:]
        else:
            self.pool_record = self.bag[:]
        self.collisions = 0
        self.moves_memory = [6 for _ in range(20)]
    def print_map(self):
        checkboard = True
        row_n = 0
        print("---"*11)
        for row in self.map:
            row_n += 1
            print((cl.Back.RED if row_n == 3 else cl.Back.RESET) + "|" + cl.Back.RESET, end="")
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
            print((cl.Back.RED if row_n == 3 else cl.Back.RESET) + "|" + cl.Back.RESET)
            checkboard = not checkboard
        print("Hold: " + self.hold_type)
        print("Score: " + str(self.score))
        print("Bag: " + str(self.bag))

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
        if not self.pocket:
            self.generate_pocket()
        ptype = self.bag[0]
        self.bag.pop(0)
        self.bag.append(self.pocket[0])
        self.pool_record.append(self.pocket[0])
        self.pocket.pop(0)
        p = layouts[ptype][0]
        ap = align_to((0,3),p)
        for dot in ap:
            if self.map[dot] != 0:
                return False
        self.current_piece = ap + [ptype]
        self.visible_current(True)
        self.rot = 0
        return True

    def move_left(self):
        self.collisions = 0
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
        self.stash(0)

    def move_right(self):
        self.collisions = 0
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
        self.stash(1)

    def soft_drop(self):
        self.collisions = 0
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
        self.stash(2)

    def rotate(self):
        self.collisions = 0
        self.visible_current(False)
        self.shift = dealign(self.current_piece, self.rot)
        for i in range(1,4):
            available = True
            rotation = self.rot + i
            while rotation > 3:
                rotation -= 4
            next = align_to(self.shift,layouts[self.current_piece[-1]][rotation]) + [self.current_piece[-1]]

            while not all([True if 0 <= x <= 9 and y <= 22 else False for y,x in next[:-1]]):
                xs = [x for y,x in next[:-1]]
                ys = [y for y,x in next[:-1]]

                if min(xs) < 0:
                    next = [(y,x-min(xs)) for y,x in next[:-1]] + [next[-1]]
                if max(xs) > 9:
                    next = [(y,x-max(xs)+9) for y,x in next[:-1]] + [next[-1]]
                if max(ys) > 22:
                    next = [(y-max(ys)+22,x) for y,x in next[:-1]] + [next[-1]]

            for dot in next[:-1]:
                if self.map[dot]:
                    available = False
                    break
            if available:
                self.current_piece = next
                self.rot = rotation
                break
        self.visible_current(True)
        self.stash(5)

    def drop(self):
        self.collisions = 0
        prev = []
        self.shift = dealign(self.current_piece, self.rot)
        while prev != self.current_piece:
            prev = self.current_piece
            self.soft_drop()
            self.score += 1
        self.clear_lines()
        if not self.create_piece():
            self.lost = True
        self.stash(3)
        self.scan_readiness()

    def clear_lines(self):
        self.shift = dealign(self.current_piece, self.rot)
        if self.current_piece[-1] == "t":
            corners_coords = align_to(self.shift,[(0,0),(0,2),(2,0),(2,2)])
            corners = [i for i in corners_coords if i[0] > 22 or i[1] < 0 or i[1] > 9 or self.map[i] != 0]
            if len(corners) >= 3:
                self.prev_comb = "tspin"
                if len(corners) == 3:
                    if (corners == [i for i in align_to(self.shift,[(0,0),(2,0),(2,2)])
                        if i[1] < 0 or i[1] > 9 or self.map[i] != 0]
                        and (self.rot == 0 or self.rot == 1)) \
                            or (corners == [i for i in align_to(self.shift,[(0,2),(2,0),(2,2)])
                                if i[1] < 0 or i[1] > 9 or self.map[i] != 0]
                                and (self.rot == 0 or self.rot == 3)) \
                            or (corners == [i for i in align_to(self.shift,[(0,0),(0,2),(2,2)])
                                if i[1] < 0 or i[1] > 9 or self.map[i] != 0]
                                and (self.rot == 3 or self.rot == 2)) \
                            or (corners == [i for i in align_to(self.shift,[(0,0),(0,2),(2,0)])
                                if i[1] < 0 or i[1] > 9 or self.map[i] != 0]
                                and (self.rot == 1 or self.rot == 2)):
                            self.prev_comb = "minitspin"
            else:
                self.prev_comb = ""
        else:
            self.prev_comb = ""

        # collision count
        for dot in self.current_piece[:-1]:
            if dot[0] == 22:
                self.collisions += 1
            else:
                if self.map[(dot[0]+1,dot[1])] and (dot[0]+1,dot[1]) not in self.current_piece:
                    self.collisions += 1

            if dot[0] > 0:
                if self.map[(dot[0]-1,dot[1])] and (dot[0]-1,dot[1]) not in self.current_piece:
                    self.collisions += 1

            if dot[1] == 0:
                self.collisions += 1
            else:
                if self.map[(dot[0],dot[1]-1)] and (dot[0],dot[1]-1) not in self.current_piece:
                    self.collisions += 1

            if dot[1] == 9:
                self.collisions += 1
            else:
                if self.map[(dot[0],dot[1]+1)] and (dot[0],dot[1]+1) not in self.current_piece:
                    self.collisions += 1

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

        score_gained = 0
        if self.prev_comb == "" and cleared == 1:
            score_gained = 100
        elif self.prev_comb == "minitspin" and cleared == 1:
            score_gained = 200
        elif self.prev_comb == "" and cleared == 2:
            score_gained = 300
        elif self.prev_comb == "minitspin" and cleared == 2:
            score_gained = 400
        elif self.prev_comb == "" and cleared == 3:
            score_gained = 500
        elif self.prev_comb == "tetris":
            score_gained = 800
        elif self.prev_comb == "tspin" and cleared == 1:
            score_gained = 800
        elif self.prev_comb == "tspin" and cleared == 2:
            score_gained = 1200
        elif self.prev_comb == "tspin" and cleared == 3:
            score_gained = 1600

        if cleared:
            if self.b2b_comb == self.prev_comb and self.prev_comb != "":
                score_gained *= 1.5
                score_gained += 50*self.b2b
                self.b2b += 1
            else:
                self.b2b = 0
            self.b2b_comb = "tspin" if self.prev_comb == "tspin" or self.prev_comb == "minitspin" else self.prev_comb
        self.score += score_gained

    def generate_7bag(self):
        self.bag = []
        standard = ["i","o","t","s","z","l","j"]
        for i in range(7):
            ptype = np.random.choice(standard)
            standard.remove(ptype)
            self.bag.append(ptype)

    def generate_pocket(self):
        self.pocket = []
        standard = ["i","o","t","s","z","l","j"]
        for i in range(7):
            ptype = np.random.choice(standard)
            standard.remove(ptype)
            self.pocket.append(ptype)

    def swap_hold(self):
        self.collisions = 0
        _bht = self.hold_type
        self.visible_current(False)
        if _bht == "":
            self.hold_type = self.current_piece[-1]
            if not self.pocket:
                self.generate_pocket()
            ptype = self.bag[0]
            self.bag.pop(0)
            self.bag.append(self.pocket[0])
            self.pool_record.append(self.pocket[0])
            self.pocket.pop(0)
            p = layouts[ptype][0]
            ap = align_to((0,3),p)
            self.current_piece = ap + [ptype]
            self.visible_current(True)
            self.rot = 0
        else:
            self.hold_type = self.current_piece[-1]
            self.current_piece = align_to((0,3),layouts[_bht][0]) + [_bht]
            self.rot = 0
            self.visible_current(True)
        self.stash(4)

    def stash(self, move):
        self.moves_memory.insert(0,move)
        self.moves_memory.pop(-1)

    def get_state(self):
        state = []
        standard = ["i", "o", "t", "s", "z", "l", "j", ""]
        combs_standard = ["","tetris","tspin"]
        # self.visible_current(False)
        state += [1 if i else 0 for i in self.map.flatten().tolist()]
        self.visible_current(True)

        tr_bag = [standard.index(i) for i in self.bag]
        state += tr_bag

        flat_coords = []
        for y, x in self.current_piece[:-1]:
            flat_coords.append(y)
            flat_coords.append(x)
        state += flat_coords

        state += [standard.index(self.hold_type)]

        state += [combs_standard.index(self.b2b_comb)]

        state += self.moves_memory

        return np.array(state)

    def scan_readiness(self):
        self.visible_current(False)
        self.lines_readiness = sum([bool(i) for i in self.map.flatten()]) / self.scan_started_lines()
        self.visible_current(True)

    def scan_started_lines(self):
        lines_started = 0
        for y in range(0,23):
            for x in range(0,10):
                if self.map[y,x]:
                    lines_started += 1
                    break
        return lines_started