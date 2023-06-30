import math
import random
import sys

import pygame
import os
import config

# dodato
import pprint
import itertools
from queue import PriorityQueue
import numpy as np
import heapq


class BaseSprite(pygame.sprite.Sprite):
    images = dict()

    def __init__(self, x, y, file_name, transparent_color=None, wid=config.SPRITE_SIZE, hei=config.SPRITE_SIZE):
        pygame.sprite.Sprite.__init__(self)
        if file_name in BaseSprite.images:
            self.image = BaseSprite.images[file_name]
        else:
            self.image = pygame.image.load(os.path.join(config.IMG_FOLDER, file_name)).convert()
            self.image = pygame.transform.scale(self.image, (wid, hei))
            BaseSprite.images[file_name] = self.image
        # making the image transparent (if needed)
        if transparent_color:
            self.image.set_colorkey(transparent_color)
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)


class Surface(BaseSprite):
    def __init__(self):
        super(Surface, self).__init__(0, 0, 'terrain.png', None, config.WIDTH, config.HEIGHT)


class Coin(BaseSprite):
    def __init__(self, x, y, ident):
        self.ident = ident
        super(Coin, self).__init__(x, y, 'coin.png', config.DARK_GREEN)

    def get_ident(self):
        return self.ident

    def position(self):
        return self.rect.x, self.rect.y

    def draw(self, screen):
        text = config.COIN_FONT.render(f'{self.ident}', True, config.BLACK)
        text_rect = text.get_rect(center=self.rect.center)
        screen.blit(text, text_rect)


class CollectedCoin(BaseSprite):
    def __init__(self, coin):
        self.ident = coin.ident
        super(CollectedCoin, self).__init__(coin.rect.x, coin.rect.y, 'collected_coin.png', config.DARK_GREEN)

    def draw(self, screen):
        text = config.COIN_FONT.render(f'{self.ident}', True, config.RED)
        text_rect = text.get_rect(center=self.rect.center)
        screen.blit(text, text_rect)


class Agent(BaseSprite):
    def __init__(self, x, y, file_name):
        super(Agent, self).__init__(x, y, file_name, config.DARK_GREEN)
        self.x = self.rect.x
        self.y = self.rect.y
        self.step = None
        self.travelling = False
        self.destinationX = 0
        self.destinationY = 0

    def set_destination(self, x, y):
        self.destinationX = x
        self.destinationY = y
        self.step = [self.destinationX - self.x, self.destinationY - self.y]
        magnitude = math.sqrt(self.step[0] ** 2 + self.step[1] ** 2)
        self.step[0] /= magnitude
        self.step[1] /= magnitude
        self.step[0] *= config.TRAVEL_SPEED
        self.step[1] *= config.TRAVEL_SPEED
        self.travelling = True

    def move_one_step(self):
        if not self.travelling:
            return
        self.x += self.step[0]
        self.y += self.step[1]
        self.rect.x = self.x
        self.rect.y = self.y
        if abs(self.x - self.destinationX) < abs(self.step[0]) and abs(self.y - self.destinationY) < abs(self.step[1]):
            self.rect.x = self.destinationX
            self.rect.y = self.destinationY
            self.x = self.destinationX
            self.y = self.destinationY
            self.travelling = False

    def is_travelling(self):
        return self.travelling

    def place_to(self, position):
        self.x = self.destinationX = self.rect.x = position[0]
        self.y = self.destinationX = self.rect.y = position[1]

    # coin_distance - cost matrix
    # return value - list of coin identifiers (containing 0 as first and last element, as well)
    def get_agent_path(self, coin_distance):
        pass


class ExampleAgent(Agent):
    def __init__(self, x, y, file_name):
        super().__init__(x, y, file_name)

    def get_agent_path(self, coin_distance):
        path = [i for i in range(1, len(coin_distance))]
        random.shuffle(path)
        return [0] + path + [0]


class Aki(Agent):
    def __int__(self, x, y, file_name):
        super().__init__(x, y, file_name)

    def __adj_nodes_cost_paths(self, coin_distance, index):
        path = [i for i in range(len(coin_distance))]
        s = []
        for i in range(len(path)):
            if path[i] != index:
                s.append(path[i])
        cost_paths = {}
        for i in range(len(s)):
            cost_paths[s[i]] = coin_distance[index][s[i]]
        return cost_paths

    def get_agent_path(self, coin_distance):
        stack = [0]
        visited = [0]
        while len(visited) != len(coin_distance):
            curr_node = stack.pop(0)
            if curr_node not in visited:
                visited.append(curr_node)
            scp = self.__adj_nodes_cost_paths(coin_distance, curr_node)
            # print("Za cvor: " + str(curr_node) + " susedi su: " + str(scp))
            scp = dict(sorted(scp.items(), key=lambda item: (item[1], item[0])))
            # print("Za cvor: " + str(curr_node) + " posle sortiranja rastuce po vrednosti susedi su: " + str(scp))
            # dodavanje sortirane liste na pocetak velike liste
            scp_keys = list(scp.keys())
            scp_final = []
            for elem in scp_keys:
                if elem not in visited:
                    scp_final.append(elem)
            stack[0:0] = scp_final
            # print("Globalna lista: " + str(stack) + " nakon dodavanja suseda cvora: " + str(curr_node))
        print("OPTIMAL PATH: " + str(visited + [0]))
        return visited + [0]


class Jocke(Agent):
    def __int__(self, x, y, file_name):
        super().__init__(x, y, file_name)

    def get_agent_path(self, coin_distance):
        # print("COIN_DISTANCE:")
        # print(coin_distance)
        path = [i for i in range(1, len(coin_distance))]
        path_all_perms = list(itertools.permutations(path, len(path)))
        min_cost = []
        mc = sys.maxsize
        for k in range(len(path_all_perms)):
            p = list(path_all_perms[k])
            perm = [0] + p + [0]
            ind = 0
            perm_cost = 0
            while ind + 1 < len(perm):
                r = perm[ind]
                c = perm[ind + 1]
                perm_cost += coin_distance[r][c]
                ind += 1
            # print("Za permutaciju: " + str(perm) + ", tezina je: " + str(perm_cost))
            if perm_cost < mc:
                mc = perm_cost
                min_cost = perm.copy()
        print("OPTIMAL PATH: " + str(min_cost))
        return min_cost


class Uki(Agent):
    def __int__(self, x, y, file_name):
        super().__init__(x, y, file_name)

    def __get_successors(self, coin_distance, partial_path: tuple):
        # path = [i for i in range(len(coin_distance))]
        # succ = []
        # for elem in path:
        #     pp = list(partial_path[-1].copy())
        #     if elem not in partial_path[-1]:
        #         pp.append(elem)
        #     if set(pp) != set(partial_path[-1]):
        #         succ.append(pp)
        # successors = []
        # for s in succ:
        #     curr_node = partial_path[-1][-1]
        #     curr_price = partial_path[0]
        #     updated_price = curr_price + coin_distance[curr_node][s[-1]]
        #     successors.append((updated_price, -len(s), s))
        # return successors
        leaf_node = partial_path[-1][-1]
        path = [i for i in range(len(coin_distance))]
        curr_price = partial_path[0]
        successors = []
        for coin in path:
            ppc = partial_path[-1].copy()
            if coin not in partial_path[-1]:
                ppc.append(coin)
                updated_price = curr_price + coin_distance[leaf_node][coin]
                successors.append((updated_price, -len(ppc), ppc[-1], ppc))
        return successors

    def get_agent_path(self, coin_distance):
        # Resenje 1 - PriorityQueue
        # pq = PriorityQueue()
        # pq.put((0, 0, [0]))
        # while not pq.empty():
        #     curr_pp = pq.get()
        #     if len(curr_pp[-1]) == len(coin_distance) + 1:
        #         print("OPTIMAL_PATH: " + str(curr_pp[-1]))
        #         return curr_pp[-1]
        #     if len(curr_pp[-1]) == len(coin_distance):
        #         curr_pp = list(curr_pp)
        #         curr_pp[0] += coin_distance[curr_pp[-1][-1]][0]
        #         curr_pp[-1].append(0)
        #         curr_pp_t = tuple(curr_pp)
        #         pq.put(curr_pp_t)
        #     curr_pp_successors = self.__get_successors(coin_distance, curr_pp)
        #     for p in curr_pp_successors:
        #         pq.put(p)

        # Resenje 2 - heapq
        # price, -nivo, poslednji identifikator na parc. putanji, parcijalna putanja
        h = []
        heapq.heapify(h)
        heapq.heappush(h, (0, 0, 0, [0]))
        while h:
            curr_pp = heapq.heappop(h)
            if len(curr_pp[-1]) == len(coin_distance) + 1:
                print("OPTIMAL_PATH: " + str(curr_pp[-1]))
                return curr_pp[-1]
            if len(curr_pp[-1]) == len(coin_distance):
                curr_pp = list(curr_pp)
                curr_pp[0] += coin_distance[curr_pp[-1][-1]][0]
                curr_pp[-1].append(0)
                # curr_pp_t = tuple(curr_pp)
                curr_pp_t = (curr_pp[0], -len(curr_pp), curr_pp[-1][-1], curr_pp[-1])
                heapq.heappush(h, curr_pp_t)
            curr_pp_successors = self.__get_successors(coin_distance, curr_pp)
            for curr_pp_suc in curr_pp_successors:
                heapq.heappush(h, curr_pp_suc)


class Micko(Agent):
    def __int__(self, x, y, file_name):
        super().__init__(x, y, file_name)

    def __update_partial_path(self, coin_distance, partial_path: list):
        path = [i for i in range(1, len(coin_distance))]
        ppn = [0]
        for elem in path:
            if elem not in partial_path:
                ppn.append(elem)
        return ppn

    def __cut_coins(self, coin_distance, partial_path: tuple):
        ppold = partial_path[-2].copy()
        if (len(coin_distance) == len(ppold)) or (len(ppold) == 1 and ppold[0] == 0):
            return coin_distance
        ppu = self.__update_partial_path(coin_distance, ppold)
        mat_out = []
        for i in range(len(coin_distance)):
            if i not in ppu:
                continue
            mat_in = []
            for j in range(len(coin_distance[i])):
                if j not in ppu:
                    continue
                mat_in.append(coin_distance[i][j])
            mat_out.append(mat_in)
        return mat_out

    def __get_prim_mst_cost(self, coin_distance, partial_path: tuple):
        coin_distance_cut = self.__cut_coins(coin_distance, partial_path)
        explored = [0] * len(coin_distance_cut)
        mst_num_of_edges = 0
        explored[0] = 1
        mst_cost = 0
        while mst_num_of_edges < len(coin_distance_cut) - 1:
            min_cost = sys.maxsize
            r = 0
            c = 0
            for i in range(len(coin_distance_cut)):
                if not explored[i]:
                    continue
                for j in range(len(coin_distance_cut)):
                    if coin_distance_cut[i][j] and (not explored[j]):
                        if coin_distance_cut[i][j] < min_cost:
                            min_cost = coin_distance_cut[i][j]
                            r = i
                            c = j
            # print(str(r) + "-" + str(c) + ":" + str(coin_distance[start][end]))
            mst_cost += coin_distance_cut[r][c]
            explored[c] = 1
            mst_num_of_edges += 1
        return mst_cost

    def __get_successors_mst(self, coin_distance, partial_path: tuple):
        # cena za sort/stvarna cena + mst, -nivo, poslednji identifikator na parc. putanji, parc. putanja, stvarna cena
        mst_cost = self.__get_prim_mst_cost(coin_distance, partial_path)
        # print("MST_COST: " + str(mst_cost))
        leaf_node = partial_path[-2][-1]
        path = [i for i in range(len(coin_distance))]
        curr_price = partial_path[-1]
        successors = []
        for elem in path:
            ppc = partial_path[-2].copy()
            if elem not in partial_path[-2]:
                ppc.append(elem)
                real_price = curr_price + coin_distance[leaf_node][elem]
                updated_price = real_price + mst_cost
                successors.append((updated_price, -len(ppc), ppc[-1], ppc, real_price))
        return successors

    def get_agent_path(self, coin_distance):
        # cena za sort/stvarna cena + mst, -nivo, poslednji identifikator na parc. putanji, parc. putanja, stvarna cena
        h = []
        heapq.heapify(h)
        heapq.heappush(h, (0, 0, 0, [0], 0))
        while h:
            curr_pp = heapq.heappop(h)
            if len(curr_pp[-2]) == len(coin_distance) + 1:
                print("OPTIMAL_PATH: " + str(curr_pp[-2]))
                return curr_pp[-2]
            if len(curr_pp[-2]) == len(coin_distance):
                curr_pp = list(curr_pp)
                curr_pp[0] = curr_pp[-1] + coin_distance[curr_pp[-2][-1]][0]
                curr_pp[-1] += coin_distance[curr_pp[-2][-1]][0]
                curr_pp[-2].append(0)
                # curr_pp_t = tuple(curr_pp)
                curr_pp_t = (curr_pp[0], -len(curr_pp), curr_pp[-2][-1], curr_pp[-2], curr_pp[-1])
                heapq.heappush(h, curr_pp_t)
            curr_pp_successors = self.__get_successors_mst(coin_distance, curr_pp)
            for curr_pp_suc in curr_pp_successors:
                heapq.heappush(h, curr_pp_suc)
