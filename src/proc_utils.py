import multiprocessing


import functools
def delayed(function):
    """Decorator used to capture the arguments of a function."""

    def delayed_function(*args, **kwargs):
        return function, args, kwargs
    try:
        delayed_function = functools.wraps(function)(delayed_function)
    except AttributeError:
        " functools.wraps fails on some callable objects "
    return delayed_function
    
    

def apply_delayed(obj):
    function, args, kwargs = obj
    return function(*args, **kwargs)

def data_reader(delayed_source, output_queue, done_output_cnt):
    # Read data in chunks

    itr = delayed_source()
    for i,data_chunk in enumerate(itr):
        if i==0:
            print(data_chunk)
        
        output_queue.put(data_chunk)
    done_output_cnt[0].value += 1


def map_process(input_queue, output_queue, map_function, done_input_cnt,done_output_cnt):
    while True:
        if done_input_cnt[0].value == done_input_cnt[1] and input_queue.empty():
            break
        data_chunk = input_queue.get()
        for processed_data in map_function(data_chunk):
            output_queue.put(processed_data)
    done_output_cnt[0].value += 1
                

import time
def run_mapping_pipeline(data_source, map_functions, num_workers=10,maxsize=None):
    
    
    num_stages = len(map_functions)

    # Queues for each stage, including the reader
    if maxsize is None:
        queues = [multiprocessing.Queue() for _ in range(num_stages+1)]
    else:
        assert isinstance(maxsize,list)  
        queues = [multiprocessing.Queue(maxsize=m) for _,m in zip(range(num_stages+1),maxsize)]
    
    done_cnt = [(multiprocessing.Value('i', 0),1)]
    for _ in range(num_stages):
        done_cnt.append((multiprocessing.Value('i', 0),num_workers))
    # Start the data reader worker
    reader_process = multiprocessing.Process(target=data_reader, args=(data_source, queues[0], done_cnt[0]))

    reader_process.start()

    # Create and start workers for each map stage
    workers = []
    for stage in range(num_stages):
        for _ in range(num_workers):
            worker = multiprocessing.Process(target=map_process, args=(queues[stage], queues[stage + 1], map_functions[stage], done_cnt[stage],done_cnt[stage+1]))
            worker.start()
            workers.append(worker)
    # def gen():
    # Collect results and handle end signals
    try:
        while done_cnt[-1][0].value < num_workers:  # Wait for all map workers to send end signal
            result = queues[-1].get()
            yield result
            print("yielded")
                
    except KeyboardInterrupt:
        for worker in workers:
            worker.terminate()
        reader_process.terminate()
        raise
    except Exception as e:
        for worker in workers:
            worker.terminate()
        reader_process.terminate()
        raise e
    finally:
        for queue in queues:
            queue.close()
            queue.join_thread()
        reader_process.join()
        for worker in workers:
            worker.join()
    # return gen