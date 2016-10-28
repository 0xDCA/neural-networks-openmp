import timeit

times_per_command = 3

def benchmark(base_command, argument_strings):
    print "Benchmarking", base_command
    return [timeit.timeit('__import__("os").system("{0} {1}")'.format(base_command, str(arg_string)),
        number=times_per_command) for arg_string in argument_strings]

def write_result_file(results, file_name):
    with open("results/" + file_name, "w") as f:
        for result in results:
            f.write("{0} {1}\n".format(str(result[0]), str(result[1])))

def full_benchmark(file_prefix, command, argument_strings, names):
    results = benchmark(command, argument_strings)

    write_result_file(zip(names, results), file_prefix + "_results.txt")

    speedup = [(float(results[0]) / item) for item in results]

    write_result_file(zip(names, speedup), file_prefix + "_speedup.txt")

thread_args = ["1 1000 100", "2 1000 100", "3 1000 100", "4 1000 100", "5 1000 100"]
thread_names = [1,2,3,4,5]

full_benchmark("nn_threads", "./Neural_networks", thread_args, thread_names)

thread_args = ["1 1000 200", "2 1000 200", "3 1000 200", "4 1000 200", "5 1000 200"]
full_benchmark("nn_threads_200", "./Neural_networks", thread_args, thread_names)

thread_args = ["1 1000 50", "2 1000 50", "3 1000 50", "4 1000 50", "5 1000 50"]
full_benchmark("nn_threads_50", "./Neural_networks", thread_args, thread_names)

thread_args = ["1 1000 150", "2 1000 150", "3 1000 150", "4 1000 150", "5 1000 150"]
full_benchmark("nn_threads_150", "./Neural_networks", thread_args, thread_names)
