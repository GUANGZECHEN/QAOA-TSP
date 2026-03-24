import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import random

class IsingModel:
    def __init__(self, h, J, const):
        self.h = h
        self.J = J
        self.const = const
        
class TSP:

    def __init__(
        self,
        N,
        distance_matrix=None,
        coordinates=None,
        is_symmetric=True,
        is_geographical=False,
        distance_metric="euclidean",
        start_city=0,
        return_to_start=True,
        seed=None,
        name=None
    ):

        self.N = N
        self.is_symmetric = is_symmetric
        self.is_geographical = is_geographical
        self.distance_metric = distance_metric
        self.start_city = start_city
        self.return_to_start = return_to_start
        self.seed = seed
        self.name = name

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.coordinates = coordinates
        self.distance_matrix = distance_matrix

        if self.is_geographical and coordinates is not None:
            self.compute_distance_matrix()
        
        # -----------------------------
        # model-related storage
        # -----------------------------
        self.Q = None
        self.ising = None

    # =========================================================
    # ensure qubo and ising representation are there
    # =========================================================

    def ensure_qubo(self):
        if self.Q is None:
            self.build_qubo()

    def ensure_ising(self):
        if self.ising is None:
            self.ensure_qubo()
            self.to_ising()
            
    # =========================================================
    # QUBO builder
    # =========================================================

    def build_qubo(self, A=None):

        N = self.N
        num_vars = N * N
        Q = np.zeros((num_vars, num_vars))

        def idx(i, t):
            return i * N + t

        if A is None:
            A = 10 * np.max(self.distance_matrix)

        # --- distance term ---
        for t in range(N):

            t_next = (t + 1) % N if self.return_to_start else t + 1
            if t_next >= N:
                continue

            for i in range(N):
                for j in range(N):
                    Q[idx(i, t), idx(j, t_next)] += self.distance_matrix[i, j]

        # --- each city once ---
        for i in range(N):
            for t1 in range(N):
                for t2 in range(N):
                    Q[idx(i, t1), idx(i, t2)] += A
            for t in range(N):
                Q[idx(i, t), idx(i, t)] -= 2 * A

        # --- one city per step ---
        for t in range(N):
            for i1 in range(N):
                for i2 in range(N):
                    Q[idx(i1, t), idx(i2, t)] += A
            for i in range(N):
                Q[idx(i, t), idx(i, t)] -= 2 * A

        self.Q = Q
        return Q
        
    # =========================================================
    # QUBO → Ising
    # =========================================================

    def to_ising(self):

        self.ensure_qubo()

        Q = np.array(self.Q)

        J = Q / 4.0
        h = -0.25 * (np.sum(Q, axis=1) + np.sum(Q, axis=0))
        const = 0.25 * np.sum(Q)

        self.ising = IsingModel(h, J, const)
        return self.ising

    # =========================================================
    # Energy functions
    # =========================================================

    def qubo_energy(self, x):
        self.ensure_qubo()
        return x @ self.Q @ x

    def ising_energy(self, z):
        self.ensure_ising()
        return z @ self.ising.J @ z + self.ising.h @ z + self.ising.const

    # =========================================================
    # Mapping utilities (CRUCIAL for solvers)
    # =========================================================

    def x_to_route(self, x):
        N = self.N
        X = x.reshape(N, N)
        route = []
        for t in range(N):
            route.append(np.argmax(X[:, t]))
        return route

    def route_to_x(self, route):
        N = self.N

        if len(route) != N:
            raise ValueError("Route must have length N")

        x = np.zeros(N * N, dtype=int)

        def idx(i, t):
            return i * N + t

        for t in range(N):
            i = route[t]
            x[idx(i, t)] = 1

        return x
    
    def z_to_x(self, z):
        return ((1 - z) // 2).astype(int)

    # =========================================================
    # Sanity check between Qubo and Ising representation
    # =========================================================

    def sanity_check_qubo_to_ising(self, atol=1e-8):

        self.ensure_ising()

        Q = self.Q
        h = self.ising.h
        J = self.ising.J
        const = self.ising.const

        n = Q.shape[0]

        for bits in itertools.product([0, 1], repeat=n):

            x = np.array(bits)
            z = 1 - 2 * x

            E_qubo = x @ Q @ x
            E_ising = z @ J @ z + h @ z + const

            if not np.isclose(E_qubo, E_ising, atol=atol):
                print("❌ mismatch")
                print("x:", x)
                print("E_qubo:", E_qubo)
                print("E_ising:", E_ising)
                raise ValueError("QUBO → Ising mismatch")

        print("🎉 QUBO → Ising transformation is EXACT!")                                
    # ---------------------------------------------------------
    # Random instance generator
    # ---------------------------------------------------------

    @classmethod
    def random_geographical(cls, N, seed=None, name="random_instance"):

        if seed is not None:
            np.random.seed(seed)

        coordinates = np.random.rand(N, 2) * 100

        return cls(
            N=N,
            coordinates=coordinates,
            is_geographical=True,
            is_symmetric=True,
            seed=seed,
            name=name
        )

    @classmethod
    def random_asymmetric(
        cls,
        N,
        min_distance=1,
        max_distance=100,
        integer_costs=True,
        seed=None,
        name="random_ATSP"
    ):

        if seed is not None:
            np.random.seed(seed)

        if integer_costs:
            matrix = np.random.randint(
                min_distance,
                max_distance,
                size=(N, N)
            ).astype(float)
        else:
            matrix = np.random.uniform(
                min_distance,
                max_distance,
                size=(N, N)
            )

        # zero diagonal (no self travel)
        np.fill_diagonal(matrix, 0)

        return cls(
            N=N,
            distance_matrix=matrix,
            is_symmetric=False,
            is_geographical=False,
            seed=seed,
            name=name
        )

    # ---------------------------------------------------------
    # Distance computation
    # ---------------------------------------------------------

    def compute_distance_matrix(self):

        if self.coordinates is None:
            raise ValueError("Coordinates required for geographical TSP")

        N = self.N
        matrix = np.zeros((N, N))

        for i in range(N):
            for j in range(N):

                if i == j:
                    continue

                x1, y1 = self.coordinates[i]
                x2, y2 = self.coordinates[j]

                if self.distance_metric == "euclidean":
                    dist = sqrt((x1 - x2)**2 + (y1 - y2)**2)

                elif self.distance_metric == "manhattan":
                    dist = abs(x1 - x2) + abs(y1 - y2)

                else:
                    raise ValueError("Unknown distance metric")

                matrix[i, j] = dist

        if self.is_symmetric:
            matrix = (matrix + matrix.T) / 2

        self.distance_matrix = matrix

    # ---------------------------------------------------------
    # Route utilities
    # ---------------------------------------------------------

    def is_valid_route(self, route):

        if len(route) != self.N:
            return False

        if set(route) != set(range(self.N)):
            return False

        return True

    def route_cost(self, route):

        if not self.is_valid_route(route):
            raise ValueError("Invalid route")

        cost = 0

        for i in range(len(route) - 1):
            cost += self.distance_matrix[route[i], route[i+1]]
            #print(cost)

        if self.return_to_start:
            cost += self.distance_matrix[route[-1], route[0]]
            #print(cost)


        return cost

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------

    def plot(self, route=None):

        N = self.N

        # --------------------------------------------------
        # choose coordinates
        # --------------------------------------------------
        if self.is_geographical:

            if self.coordinates is None:
                raise ValueError("Coordinates required for geographical plotting")

            coords = self.coordinates

        else:

            angles = np.linspace(0, 2*np.pi, N, endpoint=False)

            coords = np.array([
                (np.cos(a), np.sin(a)) for a in angles
            ])

        x = coords[:,0]
        y = coords[:,1]

        plt.figure(figsize=(6,6))
        plt.scatter(x, y, s=120)

        # city labels
        for i,(xi,yi) in enumerate(coords):
            plt.text(xi*1.07, yi*1.07, str(i), fontsize=12)

        # --------------------------------------------------
        # draw edges
        # --------------------------------------------------

        # --------------------------------------------------
        # draw edges
        # --------------------------------------------------

        if self.distance_matrix is not None:

            for i in range(N):
                for j in range(i+1, N):

                    x1,y1 = coords[i]
                    x2,y2 = coords[j]

                    cost_ij = self.distance_matrix[i,j]
                    cost_ji = self.distance_matrix[j,i]

                    xm = (x1+x2)/2
                    ym = (y1+y2)/2

                    # -----------------------------
                    # symmetric case
                    # -----------------------------

                    if self.is_symmetric:

                        plt.plot([x1,x2],[y1,y2], color="gray", alpha=0.5)

                        plt.text(
                            xm,
                            ym,
                            f"{cost_ij:.1f}",
                            fontsize=9,
                            ha="center"
                        )

                    # -----------------------------
                    # asymmetric case
                    # -----------------------------

                    else:

                        # arrow i → j
                        plt.arrow(
                            x1,y1,
                            (x2-x1)*0.8,
                            (y2-y1)*0.8,
                            head_width=0.035,
                            length_includes_head=True,
                            color="red",
                            alpha=0.7
                        )

                        # arrow j → i
                        plt.arrow(
                            x2,y2,
                            (x1-x2)*0.8,
                            (y1-y2)*0.8,
                            head_width=0.035,
                            length_includes_head=True,
                            color="blue",
                            alpha=0.7
                        )

                        # single label with colored costs
                        plt.text(
                            xm-0.02,
                            ym,
                            f"{cost_ij:.1f}",
                            color="red",
                            fontsize=9,
                            ha="right"
                        )

                        plt.text(
                            xm,
                            ym,
                            "/",
                            color="black",
                            fontsize=9,
                            ha="center"
                        )

                        plt.text(
                            xm+0.02,
                            ym,
                            f"{cost_ji:.1f}",
                            color="blue",
                            fontsize=9,
                            ha="left"
                        )


        # --------------------------------------------------
        # draw route if provided
        # --------------------------------------------------

        if route is not None:

            if not self.is_valid_route(route):
                raise ValueError("Invalid route")

            ordered = route + [route[0]] if self.return_to_start else route

            for k in range(len(ordered)-1):

                i = ordered[k]
                j = ordered[k+1]

                x1,y1 = coords[i]
                x2,y2 = coords[j]

                dx = x2 - x1
                dy = y2 - y1

                # symmetric → simple line
                if self.is_symmetric:

                    plt.plot([x1,x2],[y1,y2],
                            color="green",
                            linewidth=3)

                # asymmetric → arrow
                else:

                   plt.arrow(
                        x1, y1,
                        dx, dy,
                        head_width=0.05,
                        length_includes_head=True,
                        color="green",
                        linewidth=2
                    )


        plt.axis("equal")
        plt.title(self.name if self.name else "TSP instance")
        plt.show()

    # ---------------------------------------------------------
    # TSPLIB loader
    # ---------------------------------------------------------

    @classmethod
    def from_tsplib(cls, filename):

        coordinates = []
        name = None
        dimension = None
        edge_weight_type = None

        reading_coords = False

        with open(filename, "r") as f:

            for line in f:

                line = line.strip()

                if line.startswith("NAME"):
                    name = line.split(":")[1].strip()

                elif line.startswith("DIMENSION"):
                    dimension = int(line.split(":")[1])

                elif line.startswith("EDGE_WEIGHT_TYPE"):
                    edge_weight_type = line.split(":")[1].strip()

                elif line.startswith("NODE_COORD_SECTION"):
                    reading_coords = True
                    continue

                elif line.startswith("EOF"):
                    break

                elif reading_coords:

                    parts = line.split()
                    if len(parts) >= 3:
                        x = float(parts[1])
                        y = float(parts[2])
                        coordinates.append((x, y))

        if dimension is None:
            raise ValueError("DIMENSION not found in TSPLIB file")

        coordinates = np.array(coordinates)

        tsp = cls(
            N=dimension,
            coordinates=coordinates,
            is_geographical=True,
            is_symmetric=True,
            name=name
        )

        tsp.compute_distance_matrix()

        return tsp




