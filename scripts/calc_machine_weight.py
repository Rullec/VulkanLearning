log = r'''
{
    "msg": "total samples 1664, current samples 188, sample percent 11.3%, total cost 0.6h, avg cost 11.4s, time left 4.7h\r\n"
}
{
    "msg": "total samples 1664, current samples 134, sample percent 8.05%, total cost 0.6h, avg cost 15.9s, time left 6.8h\r\n"
}
{
    "msg": "total samples 1664, current samples 121, sample percent 7.27%, total cost 0.6h, avg cost 17.6s, time left 7.5h\r\n"
}
{
    "msg": "total samples 1664, current samples 179, sample percent 10.76%, total cost 0.6h, avg cost 11.9s, time left 4.9h\r\n"
}
{
    "msg": "total samples 1664, current samples 259, sample percent 15.56%, total cost 0.6h, avg cost 8.2s, time left 3.2h\r\n"
}
{
    "msg": "total samples 1664, current samples 334, sample percent 20.07%, total cost 0.6h, avg cost 6.4s, time left 2.4h\r\n"
}
{
    "msg": "total samples 1664, current samples 260, sample percent 15.62%, total cost 0.6h, avg cost 8.1s, time left 7.2h\r\n"
}
{
    "msg": "total samples 1664, current samples 251, sample percent 15.08%, total cost 0.6h, avg cost 8.4s, time left 3.3h\r\n"
}
{
    "msg": "total samples 1664, current samples 68, sample percent 4.09%, total cost 0.6h, avg cost 30.9s, time left 13.7h\r\n"
}
'''
import numpy as np
def handle(log):
    # the average time cost at each machine
    cost_time_lst = []
    for i in log.split("\n"):
        if i.find("msg") != -1:
            for i in i.strip().split(','):
                if i.find("avg cost") !=-1:
                    cost_time_lst.append( float(i.split()[-1][:-1]))
    num_of_machine = len(cost_time_lst)
    avg = np.mean(cost_time_lst)

    # the sampling hardness
    hardness_lst = np.linspace(1, 1.5, num_of_machine)
    cost_time_ratio_lst = cost_time_lst / avg
    print(f"cost time {cost_time_lst}")
    print(f"hardness {hardness_lst}")
    print(f"cost time ratio {cost_time_ratio_lst}")

    # bigger number,  better perf
    perf_lst =  [1.0 / (cost_time_ratio_lst[i] / hardness_lst[i]) for i in range(num_of_machine)]
    perf_mean = np.mean(perf_lst)
    perf_ratio_lst = perf_lst / perf_mean
    perf_ratio_lst = 0.5 * (perf_ratio_lst - 1) + 1
    print(f"sampling ratio list {perf_ratio_lst}")
    


if __name__ == "__main__":
    handle(log)