from vis_nav_game import Player, Action, Phase
import pygame
import cv2
import numpy as np
import os
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from pathlib import Path
import matplotlib.pyplot as plt
import heapq

class KeyboardPlayerPyGame(Player):
    def __init__(self):
        super().__init__()
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        self.save_dir = "exploration_data/images_subsample/"
        self.sift_features = None
        self.codebook = None
        self.extractor = cv2.SIFT_create()
        self.vlad_db = None
        self.db_names = None
        self.target_images = None
        self.graph = None
        self.tree = None
        self.image_positions = {}  # map image names to (x, y)

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        pygame.init()
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT,
        }

    def set_target_images(self, images):
        self.target_images = images
        self.show_target_images()

    def get_target_images(self):
        return self.target_images

    def extracting_sift(self):
        all_descriptors = []
        for img_name in tqdm(os.listdir(self.save_dir)):
            img = cv2.imread(self.save_dir + img_name)
            kp, des = self.extractor.detectAndCompute(img, None)
            if des is not None and len(des) > 0:
                all_descriptors.extend(des)
        return np.asarray(all_descriptors)

    def create_codebook(self):
        self.sift_features = self.extracting_sift()
        self.codebook = KMeans(n_clusters=16, init="k-means++", n_init=2, verbose=0).fit(self.sift_features)

    def get_vlad(self, X, codebook):
        predictedLabels = codebook.predict(X)
        centroids = codebook.cluster_centers_
        k, d = centroids.shape
        VLAD_feature = np.zeros((k, d))
        for i in np.unique(predictedLabels):
            indices = np.where(predictedLabels == i)
            VLAD_feature[i] = np.sum(X[indices] - centroids[i], axis=0)
        VLAD_feature = VLAD_feature.flatten()
        VLAD_feature = np.sign(VLAD_feature) * np.sqrt(np.abs(VLAD_feature))
        norm = np.linalg.norm(VLAD_feature)
        if norm != 0:
            VLAD_feature /= norm
        return VLAD_feature

    def compute_vald(self):
        self.vlad_db = []
        self.db_names = []
        for img_name in tqdm(os.listdir(self.save_dir)):
            img = cv2.imread(self.save_dir + img_name)
            kp, des = self.extractor.detectAndCompute(img, None)
            if des is None or len(des) == 0:
                continue
            VLAD = self.get_vlad(des, self.codebook)
            self.vlad_db.append(VLAD)
            self.db_names.append(img_name)
        self.vlad_db = np.asarray(self.vlad_db)
        self.tree = KDTree(self.vlad_db, leaf_size=40)
        self.graph = self.compute_global_graph(neighbor_k=10)

    def compute_global_graph(self, neighbor_k=10):
        graph = {}
        for i, descriptor in enumerate(self.vlad_db):
            descriptor = descriptor.reshape(1, -1)
            distances, indices = self.tree.query(descriptor, k=neighbor_k+1)
            neighbors = [(int(idx), float(d)) for d, idx in zip(distances[0][1:], indices[0][1:])]
            graph[i] = neighbors
        return graph

    def find_path_from_fpv(self):
        print("Attempting improved path from FPV to target...")
        if self.fpv is None or self.graph is None or self.target_images is None:
            print("FPV, graph, or targets not ready.")
            return

        start_candidates = self.query([self.fpv], return_indices=True)[0]
        goal_candidates = self.query(self.target_images, return_indices=True)[0]

        print(f"Start candidates: {start_candidates}")
        print(f"Goal candidates: {goal_candidates}")

        if not start_candidates or not goal_candidates:
            print("No visual matches found for start or goal. Cannot compute path.")
            return

        best_path = None
        best_cost = float('inf')

        print(f"Trying all {len(start_candidates)}×{len(goal_candidates)} combinations...")
        for start in start_candidates:
            for goal in goal_candidates:
                print(f"Trying path from {start} → {goal}...")
                path = self.dijkstra_shortest_path(self.graph, start, goal)
                if path:
                    vlad_path = [self.vlad_db[i] for i in path]
                    total_dist = sum(np.linalg.norm(vlad_path[i] - vlad_path[i+1]) for i in range(len(vlad_path)-1))
                    if total_dist < best_cost:
                        best_cost = total_dist
                        best_path = path

        if not best_path:
            print("No path found between FPV and target candidates.")
            return

        print("Best path:", best_path)
        images = []
        for idx in best_path:
            img_path = os.path.join(self.save_dir, self.db_names[idx])
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (150, 150))
                images.append(img)

        cols = 6
        rows = (len(images) + cols - 1) // cols
        grid_rows = []
        for i in range(rows):
            row_imgs = images[i*cols:(i+1)*cols]
            while len(row_imgs) < cols:
                row_imgs.append(np.zeros_like(row_imgs[0]))
            grid_rows.append(cv2.hconcat(row_imgs))
        full_grid = cv2.vconcat(grid_rows)
        cv2.imshow("Improved Path Images", full_grid)
        cv2.waitKey(1)

    def dijkstra_shortest_path(self, graph, start, goal):
        queue = [(0, start, [start])]
        visited = set()
        while queue:
            cost, node, path = heapq.heappop(queue)
            if node == goal:
                return path
            if node in visited:
                continue
            visited.add(node)
            for neighbor, weight in graph.get(node, []):
                if neighbor not in visited:
                    heapq.heappush(queue, (cost + weight, neighbor, path + [neighbor]))
        return None


    def query(self, target, return_indices=False):
        result_list = []
        query_image_list = []
        match_indices = []
        tree = KDTree(self.vlad_db, leaf_size=40)
        result_path = Path(self.save_dir)

        for query_img in target:
            query_image_list.append(query_img)
            q_kp, q_des = self.extractor.detectAndCompute(query_img, None)
            if q_des is None or len(q_des) == 0:
                result_list.append([""])
                match_indices.append([])
                continue
            query_VLAD = self.get_vlad(q_des, self.codebook).reshape(1, -1)
            dist, indices = tree.query(query_VLAD, k=5)
            top_5_names = [self.db_names[i] for i in indices[0]]
            result_list.append(top_5_names)
            match_indices.append(indices[0].tolist())
            print(f"Query image matched indices: {indices[0].tolist()}")

        if len(query_image_list) == 0:
            print("No query images provided.")
            return [[]] if return_indices else []

        fig, axs = plt.subplots(6, len(query_image_list), figsize=(3 * len(query_image_list), 18))
        if len(query_image_list) == 1:
            axs = np.expand_dims(axs, axis=1)

        for i, query_img in enumerate(query_image_list):
            img_rgb = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
            axs[0, i].imshow(img_rgb)
            axs[0, i].axis('off')
            axs[0, i].set_title("Query")
            for j in range(5):
                result_name = result_list[i][j]
                if result_name == "":
                    axs[j + 1, i].axis('off')
                    axs[j + 1, i].set_title("No Match")
                    continue
                result_img = cv2.imread(str(result_path / result_name))
                result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                axs[j + 1, i].imshow(result_img_rgb)
                axs[j + 1, i].axis('off')
                axs[j + 1, i].set_title(f"Rank {j+1}")

        plt.tight_layout()
        # plt.show()  # Commented out to avoid blocking behavior

        if return_indices:
            return match_indices


    def show_target_images(self):
        targets = self.get_target_images()
        if not targets:
            return
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])
        h, w = concat_img.shape[:2]
        color = (0, 0, 0)
        concat_img = cv2.line(concat_img, (int(w / 2), 0), (int(w / 2), h), color, 2)
        concat_img = cv2.line(concat_img, (0, int(h / 2)), (w, int(h / 2)), color, 2)
        cv2.imshow('KeyboardPlayer:target_images', concat_img)
        cv2.waitKey(1)

    def act(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            if event.type == pygame.KEYDOWN:
                if event.key in self.keymap:
                    self.last_act |= self.keymap[event.key]
                else:
                    self.show_target_images()
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]
        return self.last_act

    def pre_exploration(self):
        print(f'K={self.get_camera_intrinsic_matrix()}')

    def pre_navigation(self):
        self.create_codebook()
        self.compute_vald()
        self.build_image_positions_from_names()
        targets = self.get_target_images()
        if targets is None:
            print("Warning: Target images not set yet, skipping query.")
            return
        self.query(targets)

    def build_image_positions_from_names(self):
        for name in self.db_names:
            try:
                x = int(name.split("_")[0][1:])
                y = int(name.split("_")[1][1:].split(".")[0])
                self.image_positions[name] = (x, y)
            except:
                continue

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return
        self.fpv = fpv
        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        def convert_opencv_img_to_pygame(opencv_image):
            opencv_image = opencv_image[:, :, ::-1]
            shape = opencv_image.shape[1::-1]
            return pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

        pygame.display.set_caption("KeyboardPlayer:fpv")
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()

        if self._state:
            if self._state[1] == Phase.NAVIGATION:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_q]:
                    self.query(self.get_target_images())
                if keys[pygame.K_f]:
                    self.find_path_from_fpv()

if __name__ == "__main__":
    import logging
    logging.basicConfig(filename='vis_nav_player.log', filemode='w', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    import vis_nav_game as vng
    logging.info(f'player.py is using vis_nav_game {vng.core.__version__}')
    vng.play(the_player=KeyboardPlayerPyGame())
