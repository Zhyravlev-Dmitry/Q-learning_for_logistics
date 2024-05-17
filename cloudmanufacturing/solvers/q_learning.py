import numpy as np
import random
from cloudmanufacturing.validation import objvalue, construct_delta

class QLearningSolver:
    def __init__(self, dataset):
        self.problems = dataset
        self.total_cost = 0
        self.service = 0  # a special case with a single provider

    def linear(self, st, end, duration, t):  # линейная интерполяция между двумя значениями st и end
        if t >= duration:
            return end
        return st + (end - st) * (t / duration)

    # q-функция показывает вознаграждение, которое мы можем получить, совершив действие.
    # Например, q[i][current_city][next_city] показывает доступное вознаграждение, если мы находимся в позиции
    # [i][current_city], если пойдем в next_city
    def random_initialization(self, available_operations, cost_operations, n_cities):
        shape = (len(available_operations), n_cities, n_cities)
        max_value = np.max(np.where(cost_operations != np.inf, cost_operations, 0))
        # q_function = np.random.uniform(10*max_value, 20 * max_value, size=shape)
        q_function = np.ones(shape)*10000000000
        return q_function

    # Размерность q-функции равна state на action, а state определяется не только (суб)операцией,
    # но и городом, в котором мы находимся!

    # Инициализируется правильно
    def greedy_initialization(self, available_operations, cost_operations, trans_cost, n_cities):
        q_function = self.random_initialization(available_operations, cost_operations, n_cities)
        for i, stage in enumerate(available_operations):
            if i == 0:
                next_city = np.argmin(cost_operations[stage])
                q_function[i, :, next_city] = cost_operations[stage][next_city]
            else:
                cost_total = cost_operations[stage] + trans_cost[:,next_city,:]
                service = 0
                current_city = next_city
                next_city = np.argmin(cost_total)
                q_function[i][current_city][next_city] = cost_total[service, next_city]

        return q_function
    def train_operaion(self, available_operations, cost_operations, time_cost, trans_cost, n_cities,
                       operation_number, gamma, delta, problem,
                       alpha=0.1, rl_gamma=0.1, epsilon=0.15, epoch_number=100000, q_function = ''):
        time_inf = np.max(time_cost)
        """
        Solve problem for one suboperation  # maybe,operation?
        """
        if isinstance(q_function, str):
            q_function = self.random_initialization(available_operations, cost_operations, n_cities)
            # q_function = self.greedy_initialization(available_operations, cost_operations, trans_cost, n_cities)


        # train
        for epoch in range(epoch_number):
            # epsilon = self.linear(epsilon, epsilon / 4, 0.5 * epoch_number, epoch)

            for i, stage in enumerate(available_operations):
                available_cities = np.where(problem["time_cost"][stage] < time_inf)[0] # почему ["time_cost"][1]?
                if i == 0:
                    if random.random() < epsilon:
                        next_city = np.random.choice(available_cities)
                    else:
                        # первый город может быть любым
                        # ошибкоопасное место, непонятно, что возвращает unravel_index
                        next_city = np.unravel_index(np.argmin(q_function[i]), q_function[i].shape)[1]

                    r = cost_operations[stage][next_city]
                    q_function[i, :, next_city] = (1-alpha) * q_function[i, :, next_city] \
                                              + alpha * (r + rl_gamma * np.argmin(q_function[i+1][next_city]))
                else:
                    service = self.service  # a special case with a single provider
                    current_city = next_city
                    cost_total = cost_operations[stage] + trans_cost[:, current_city, :]

                    if random.random() < epsilon:
                        next_city = np.random.choice(available_cities)
                    else:
                        next_city = np.argmin(q_function[i][current_city])

                    r = cost_total[service, next_city]

                    if i != len(available_operations) - 1:
                        q_function[i][current_city][next_city] = \
                            (1 - alpha) * q_function[i][current_city][next_city] \
                            + alpha * (r + rl_gamma * np.argmin(q_function[i+1][next_city]))
                    else:
                        q_function[i][current_city][next_city] = \
                            (1 - alpha) * q_function[i][current_city][next_city] \
                            + alpha * r
            # validation
            step_validation = epoch_number // 10
            if epoch % step_validation == 0:
                gamma_copy = np.copy(gamma)
                path, sum_q_function, gamma_copy, delta = self.test_operaion(
                    q_function, available_operations,
                    operation_number, gamma_copy, delta, cost_operations, trans_cost
                )
                #print('operation_number', operation_number, 'step', epoch // step_validation,
                #      'sum_q_function', sum_q_function)
        return q_function

    def test_operaion(
            self, q_function, available_operations,
            operation_number, gamma, delta, cost_operations, trans_cost
    ):
        """
        Solve problem for one suboperation
        """
        sub_problem_data = []
        cost = 0
        for i, stage in enumerate(available_operations):
            if i == 0:
                # ошибкоопасное место
                current_city, next_city = np.unravel_index(np.argmin(q_function[i]), q_function[i].shape)

                sub_problem_data.append([next_city, q_function[i][current_city][next_city]])

                cost += cost_operations[stage][next_city]

            else:
                # Here we calculate the min value of Q matrix

                service = self.service
                next_city = np.argmin(q_function[i][current_city])

                sub_problem_data.append([next_city, q_function[i, current_city, next_city]])
                delta[service, current_city, next_city, current_stage, operation_number] = 1

                cost_total = cost_operations[stage] + trans_cost[:, current_city, :]
                cost += cost_total[service, next_city]

            current_city = next_city
            current_stage = stage
            gamma[stage, operation_number, next_city] = 1

        #print('cost', cost)
        return (
            np.array(sub_problem_data)[:, 0],
            np.sum(np.array(sub_problem_data)[:, 1]),
            gamma, delta
        )

    def solve_problem(self, num_problem, alpha=0.1, rl_gamma=0.1, epsilon=0.15, epoch_number=100000):

        n_cities = self.problems[num_problem]["n_cities"]
        n_services = self.problems[num_problem]["n_services"]
        n_suboperations = self.problems[num_problem]["n_suboperations"]
        n_operations = self.problems[num_problem]["n_operations"]
        operations = self.problems[num_problem]["operations"]
        dist = self.problems[num_problem]["dist"]
        time_cost = self.problems[num_problem]["time_cost"]
        op_cost = self.problems[num_problem]["op_cost"]
        productivity = self.problems[num_problem]["productivity"]
        transportation_cost = self.problems[num_problem]["transportation_cost"]
        problem = self.problems[num_problem]

        check_point_dist = 1000
        list_q_function = []
        cost_in_check_point = []
        for i in range(epoch_number // check_point_dist):
            print(f'{i} из {epoch_number // check_point_dist}')
            # Create cost matrices
            cost_operations = time_cost * op_cost / productivity
            trans_cost = dist[None, ...] * transportation_cost

            gamma = np.zeros((n_suboperations, n_operations, n_cities))
            delta = np.zeros((n_services, n_cities, n_cities,
                              n_suboperations - 1, n_operations))

            problem_cost = 0
            problem_path = {}
            for n_sub in range(n_operations):
                available_operations = np.nonzero(operations[:, n_sub])[0]

                gamma_copy = np.copy(gamma)
                if len(list_q_function) < n_operations:
                    q_function_operation = self.train_operaion(
                        available_operations, cost_operations, time_cost, trans_cost, n_cities,
                        n_sub, gamma, delta, problem,
                        alpha, rl_gamma, epsilon, epoch_number)
                    list_q_function.append(q_function_operation)
                else:
                    q_function_operation = self.train_operaion(
                        available_operations, cost_operations, time_cost, trans_cost, n_cities,
                        n_sub, gamma, delta, problem,
                        alpha, rl_gamma, epsilon, epoch_number, list_q_function[n_sub])
                    list_q_function[n_sub] = q_function_operation


                assert np.array_equal(gamma, gamma_copy), "Портится gamma"
                path, cost, gamma, delta = self.test_operaion(
                    q_function_operation, available_operations,
                    n_sub, gamma, delta, cost_operations, trans_cost
                )

                # problem_cost += cost
                problem_path[f"suboperation_{n_sub}"] = path

            problem_cost = objvalue(self.problems[num_problem], gamma, delta)
            cost_in_check_point.append(problem_cost)
        return {"path": problem_path, "cost": problem_cost}, gamma, delta, cost_in_check_point
