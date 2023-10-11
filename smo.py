import math
import random
import multiprocessing

import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt


class QueueingSystem:
    t_A = t_D = t = Tp = N_A = N_D = n = number_clients = 0

    A = []
    D = []
    V = []
    Data = []

    def __init__(self, work_time, intensity):
        self.work_time = work_time
        self.intensity = intensity
        self.data_for_output = {}
        self.generation_func_intens = []


    def _exponential(self, intensity: float) -> float:
        U = random.uniform(0, 1)
        return ( -math.log(U) / intensity ) #if intensity else 0


    def _function_intensity(self, t: int) -> float:
        #return ( 1 + ( 2 / (t + 1) ) )
    
        if t < 1:
            return ( 0.1 * t + 5.8 ) / 59
        elif t >= 1 and t < 9:
            return t / 4
        elif t >= 9 and t < 11:
            return  ( 36 - 0.1 * t ) / 60
        else:
            return ( 54 - 0.2 * t ) / 60


    def _poisson(self, t, intensity):
        while(True):
            U1 = random.uniform(0, 1)
            t -= ( math.log(U1) / intensity ) if intensity else 0
            U2 = random.uniform(0, 1)
            if U2 <= ( self._function_intensity(t) / intensity ):
                return t

    def _enter(self):
        self.t = self.t_A
        self.N_A += 1
        self.n += 1
        self.t_A = self._poisson(self.t, self.intensity)
        if self.n == 1:
            Y = self._exponential(self.intensity)
            self.t_D = self.t + Y
            self.V.append(Y)
        
        self.A.append(self.t)
        self.Data.append(
            {
                'time': self.t,
                'queue_length': self.n,
                'client_id': self.N_A,
                'type': 'enter'
            }
        )
        self.number_clients += 1

    def _exit(self):
        self.t = self.t_D
        self.N_D += 1
        self.n -= 1
        Y = self._exponential(self.intensity)
        if self.n == 0:
            self.t_D = math.inf
        else:
            self.t_D = self.t + Y
            self.V.append(Y)
        
        self.D.append(self.t)
        self.Data.append(
            {
                'time': self.t,
                'queue_length': self.n,
                'client_id': self.N_D,
                'type': 'leave'
            }
        )

    def _last(self):
        self.t = self.t_D
        self.N_D += 1
        self.n -= 1
        if self.n > 0:
            Y =  self._exponential(self.intensity)
            self.t_D = self.t + Y
            self.V.append(Y)

        self.D.append(self.t)
        self.Data.append(
            {
                'time': self.t,
                'queue_length': self.n,
                'client_id': self.N_D,
                'type': 'leave'
            }
        )

    def _end(self):
        self.Tp = max(self.t - self.work_time, 0)

    def _calculate_metrics(self):
        time_in_queue = 0.0
        clients_in_queue = 0.0
        time_busy = 0.0

        for event in self.Data:
            if event['type'] == 'enter':
                if event['queue_length'] <= 1:
                    begin_work = event['time']
                if event['queue_length'] > 1:
                    time_in_queue += self.D[event['client_id'] - 2] - self.A[event['client_id'] - 1]
            
            if event['type'] == 'leave':
                if event['queue_length'] == 0:
                    end_work = event['time']
                    time_busy += (end_work - begin_work)

            clients_in_queue += event['queue_length']
        
        mean_time_in_queue = time_in_queue / self.number_clients 
        mean_clients_in_queue = clients_in_queue / self.number_clients
        mean_busy = time_busy / self.work_time
        mean_time_client_in_system = np.mean(np.array(self.D) - np.array(self.A))

        self.data_for_output = {
            'mean_time_in_queue': mean_time_in_queue,
            'mean_clients_in_queue': mean_clients_in_queue,
            'mean_busy': mean_busy,
            'mean_time_client_in_system': mean_time_client_in_system,
            'time_exit_last_client': self.Tp
        }  

    def _data_output(self, output_file):
        table1 = PrettyTable(['Клиент', 'Событие', 'Время события', 'В очереди'])
        for event in self.Data:
            table1.add_row([event['client_id'], event['type'], event['time'], event['queue_length']])

        table2 = PrettyTable(['Номер клиента', 'Время прихода Ai', 'Время ухода Di', 'Время обслуж. Vi', 'Время в очереди Wi', 'Время клиента в системе Ai - Di'])
        
        for i in range(len(self.A)):
            table2.add_row([i + 1, self.A[i], self.D[i], self.V[i], round(abs(self.D[i] - self.A[i] - self.V[i]), 5), self.D[i] - self.A[i]])

        metrics = f"""
        Среднее задержка клиентов в очереди: {self.data_for_output['mean_time_in_queue']}
        Среднее число клиентов в очереди: {self.data_for_output['mean_clients_in_queue']}
        Оценка занятости устройства: {self.data_for_output['mean_busy']}
        Среднее время клиента в системе: {self.data_for_output['mean_time_client_in_system']}
        Время когда ушел последний клиент: {self.data_for_output['time_exit_last_client']}
        """

        if output_file:
            with open('output.txt', 'w') as output:
                output.write(str(table1))
                output.write('\n')
                output.write(str(table2))
                output.write('\n')
                output.write(metrics)
        else:
            print(table1)
            print()
            print(table2)
            print()
            print(metrics)
  

    def simulation(self, print_data=False, output_file=False):
 
        self.t_A = self._exponential(self.intensity)
        self.t_D = math.inf

        while(True):

            self.generation_func_intens.append(
                {
                    'gen': self._function_intensity(self.t) / self.intensity,
                    't': self.t
                }
            )
            
            if (self.t_A <= self.t_D) and (self.t_A <= self.work_time):
                self._enter()
            if (self.t_D < self.t_A) and (self.t_D <= self.work_time):
                self._exit()
            if (min(self.t_A, self.t_D) > self.work_time) and (self.n > 0):
                self._last()
            if (min(self.t_A, self.t_D) > self.work_time) and (self.n == 0):
                self._end()
                break

        self._calculate_metrics()

        if print_data:
            self._data_output(output_file)

    
    def _hist_clients_in_queue(self, times, queue_length):
        #plt.figure(figsize=(20, 5))
        plt.plot(times, queue_length)
        plt.xlabel('Время')
        plt.ylabel('Количество клиентов')
        plt.title('Количество клиентов в очереди')
        plt.show()
       
    def show_hist_clients_in_queue(self):
        times = []
        queue_length = []

        for event in self.Data:
            times.append(event['time'])
            queue_length.append(event['queue_length'])

        p = multiprocessing.Process(target=self._hist_clients_in_queue, args=(times, queue_length))
        p.start()


    def _show_gen_func_intens(self, times, values):
        plt.plot(times, values)
        plt.xlabel('Время')
        plt.ylabel('λ(t)/t')
        plt.title('Генерация λ(t)/t')
        plt.show()

    def show_gen_func_intens(self):
        times = []
        values = []

        for el in self.generation_func_intens:
            times.append(el['t'])
            values.append(el['gen'])

        p = multiprocessing.Process(target=self._show_gen_func_intens, args=(times, values))
        p.start()


if __name__ == "__main__":
    INTENSITY = 5
    TIME = 12 # с 9 до 21

    smo = QueueingSystem(TIME, INTENSITY)
    smo.simulation(print_data=True, output_file=True)
    smo.show_hist_clients_in_queue()
    smo.show_gen_func_intens()

    n = 50
    sum_Tp = 0.0
    sum_St = 0.0
    for i in range(n):
        smo = QueueingSystem(TIME, INTENSITY)
        smo.simulation()
        sum_Tp += smo.Tp
        sum_St += smo.data_for_output['mean_time_client_in_system']

    print('\nКоличество прогонов: ', n)
    print('E[Tp]: ', sum_Tp / n)
    print('E[St]: ', sum_St / n)

    