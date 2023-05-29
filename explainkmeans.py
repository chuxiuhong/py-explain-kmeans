import numpy as np

np.random.seed(0)

def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)


class ExplainKmeansTree:
    def __init__(self, delta=0.2):
        # contains centers
        self.centers = []
        self.left, self.right = None, None
        self.delta = delta
        self.d = 0
        self.left_threshold = 0
        self.right_threshold = 0

    def _divide_and_share(self, theta, epsilon, delta, d):
        if len(self.centers) < 2:
            return True
        median_center = self.centers[np.abs(self.centers[:, d] - np.median(self.centers[:, d], axis=0)).argmin()]
        max_dist = euclidean_distance(median_center, self.centers[0])
        for p in self.centers:
            dis = euclidean_distance(median_center, p)
            max_dist = dis if dis > max_dist else max_dist
        left_threshold = median_center[d] + (delta + epsilon) * np.sqrt(theta) * max_dist
        right_threshold = median_center[d] + (delta - epsilon) * np.sqrt(theta) * max_dist
        left_centers = []
        right_centers = []
        for c in self.centers:
            if c[d] <= left_threshold:
                left_centers.append(c)
            if c[d] >= right_threshold:
                right_centers.append(c)
        if len(left_centers) > 0 and len(right_centers) > 0:
            self.left = ExplainKmeansTree()
            self.right = ExplainKmeansTree()
            self.left.centers = np.array(left_centers)
            self.right.centers = np.array(right_centers)
            self.d = d
            self.left_threshold = left_threshold
            self.right_threshold = right_threshold
            return True
        else:
            return False

    def expand(self):
        if len(self.centers) < 2:
            return True
        split_res = False
        while not split_res:
            d = np.random.randint(0, self.centers[0].shape[0])
            theta = np.random.random()
            epsilon = min(self.delta / (15 * np.log(self.centers[0].shape[0])), 1 / 384)
            split_res = self._divide_and_share(theta, epsilon, self.delta, d)
        self.left.expand()
        self.right.expand()

    def traverse(self,data:np.ndarray):
        print(
            f"dimension = {self.d},left value = {self.left_threshold * (np.max(data[:,self.d]) - np.min(data[:,self.d])) + np.min(data[:,self.d])}"
            f", right value = {self.right_threshold * (np.max(data[:,self.d]) - np.min(data[:,self.d])) + np.min(data[:,self.d])}"
            f", centers = {self.centers}")
        if self.left is None:
            print("-")
        else:
            self.left.traverse(data)
        if self.right is None:
            print("-")
        else:
            self.right.traverse(data)
