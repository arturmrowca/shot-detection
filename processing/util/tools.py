'''
Created on 09.08.2017

@author: Q416435
'''
import collections
import time
import sys
from multiprocessing import Process, Queue, Pool

from decorator import contextmanager
from scipy.stats import itemfreq
import matplotlib.pyplot as plt
from scipy._lib.six import xrange
import multiprocessing
from copy import deepcopy
import os
import pandas as pd
import numpy
from pymining import itemmining
import pandas
import re



global __DONE_NUMBER, __NEW_COL_THRSHLD, __LOCK_1, __KNOWN_DICT_CNT, __KNOWN_DICT

class PNT:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

class Keep(object):
    DONE_NUMBER = 0
    NEW_COL_THRSHLD = -1
    MISTER_LOCK = multiprocessing.Lock()


global TRANSACTIONS, ITEMSETS, FINAL_ROWS, FINAL_FOUND
def squash(df):
    global TRANSACTIONS 
    TRANSACTIONS.append(tuple(df["Signalname"]))
    df["collected"] = str(tuple(df["Signalname"]))
        
    return df.iloc[-1]

def verifyTSCBN(tscbn):
    """Verifies if the given tscbn is consistent"""
    nw_valid = True

    # find correct value sets
    store_tv = dict()
    for v in tscbn.V:
        if str.startswith(v, "dL_"): continue
        # check if all values of my parents are existing as condition
        tv = "_".join(v.split("_")[:-1])
        vals = []
        if tv in store_tv:
            vals = store_tv[tv]
        else:
            for i in range(0, 10000):
                node = tv + str("_") + str(i)
                if not node in tscbn.Vdata:break


                vals += tscbn.Vdata[node]["vals"]
            vals = set(list(vals))
            if not tv in store_tv:
                store_tv[tv] = vals


    for v in tscbn.V:
        if str.startswith(v, "dL_"): continue
        tv = "_".join(v.split("_")[:-1])
        if tscbn.Vdata[v]["parents"] is None:
            print("no parents for %s" % str(v))
            continue # all good



        # check if my vertex has those values:
        if len(store_tv[tv]) != len(tscbn.Vdata[v]["vals"]):
            print("Vertex %s has vals %s instead of %s" % (str(v), str(tscbn.Vdata[v]["vals"]), str(vals)))
            nw_valid = False


        # check if my children have me as parent
        for c in tscbn.Vdata[v]["children"]:
            if v not in tscbn.Vdata[c]["parents"]:
                print("Error Child %s does not contain %s in its parent set" % (str(c), str(v)))
                nw_valid = False

        # check if parent has me as child
        combos_expected = 1
        for p in tscbn.Vdata[v]["parents"]:
            if v not in tscbn.Vdata[p]["children"]:
                print("Error Parent %s does not contain %s in its children set" % (str(p), str(v)))
                nw_valid = False
            else:
                # if it has me as child needs to have dL me also as child
                if "dL_" + v not in tscbn.Vdata[p]["children"]:
                    print("Error parent of %s -> %s does not contain temporal node %s in its children set" % (str(v), str(p), str("dL_" + v)))
                    nw_valid = False

            # validate cprob -> use number of combinations
            tv = "_".join(v.split("_")[:-1])
            combos_expected *= len(store_tv[tv])

        #combos actual
        if tscbn.Vdata[v]["parents"]:
            combos_actual = len(tscbn.Vdata[v]["cprob"])
            if combos_actual != combos_expected:
                print("Vertex %s: expected combos %s -> found combos %s" % (str(v), str(combos_expected), str(combos_actual)))
                nw_valid = False

    return nw_valid




@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

@contextmanager
def suppress_nothing():

    with open(os.devnull, "w") as devnull:
        try:
            yield
        finally:
            pass

def int_str_align(obj, test_key):
    """
    This method tests if object key is string if not makes it integer and vice versa
    Then, it returns the key that is valid
    :param obj:
    :param test_key:
    :return:
    """
    try:
        a = obj[test_key]
        valid_key = test_key
    except:
        if isinstance(test_key, str):
            try:
                valid_key = int(test_key)
            except:
                valid_key = str(test_key)
        else:
            valid_key = str(test_key)
    return valid_key

def run_function_in_process(function, in_args=(1,2,3)):
    '''
    This method starts the given function with its arguments in a seperate process
    :param function:
    :param in_args: possibly needs to be [[arg1]]
    :return:
    '''
    from multiprocessing import Process, Queue
    queue = Queue()
    p = Process(target=function, args=in_args)
    p.start()
    #p.join() # this blocks until the process terminates
    #result = queue.get()


@contextmanager
def suppress_stderr():

    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

def only_valid_sequence_extension():
    nodes = ["A", "B", "C", "D", "E"]
    dest = 5 # 7 Knoten
    # Ziel
    # key: 0: A   1: A,B    2: A,B,C    3: A,B,C    4: A,B,C    5: B,C      6:  C
    # komme von links: Fill A   A,B     A,B,C
    # komme von rechts: Fill C  B,C     A,B,C       Rest in der Mitte ist A,B,C
    # bis Anzahl an nevers - so dass immer pro Knoten maximal Anzahl an Nevers e.g. wenn ich 2 never hab gibts 2 Varianten
    # dann wechseln halt die Varianten e.g. wenn ein never vergeben wird AB BC CD DE
    # Bsp 2:
    # ABCDEF 6 auf 7
    # A     A,B     B,C     C,D     D,E     E,F     F
    # ABCDE  5 auf 7 - 2 never
    # A     A,B     A,B,C       B,C,D       C,D,E   D,E     E

    res = dict()
    nevs = dest - len(nodes) # max size is nevs + 1 - when reached add 1 per iteration

    # komme von links: Fill A   A,B     A,B,C bis Anzahl never dann shifte eins rechts
    shift_idx = 0
    for i in range(1, dest+1):
        k = i
        if k > len(nodes): # k - shift_idx is size of result
            k = len(nodes) # dann ziehe immer den vordersten weg
            if dest - i+1 <  k- shift_idx:  # Anzahl an verbleibenden Stellen im Vgl. was uebrig is zum abbau
                shift_idx += 1

        size = k - shift_idx
        if size > nevs+1: # then shift index
            shift_idx += 1

        if not i in res:
            res[i] = nodes[shift_idx:k]

def plot_interval_sequence(output_interval_list):
    # All never occurrences are "invisible"
    # Sequence supposed to be t=20, sig="BedLicht", val='Aus_bed'   not BedLicht_1 etc. - because I do not know

    iq = 0
    outputs = []
    in_seq = []
    for output_intervals in output_interval_list:
        iq += 1

        x_start = []
        x_stop = []
        y = []
        label = []
        for otto in output_intervals:
            lst_seq = output_intervals[otto]
            for l in lst_seq:
                x_start += [l[1]]
                x_stop += [l[2]]
                y += [otto]
                label += [l[0]]

        #IntervalPlotter().plot_timeline(x_start, x_stop, y, label, y_shift=0.08, title="Sample %s" % str(iq))
        plt.show()

    return outputs, in_seq

def sequences_to_intervals_keep_same(random_samples, vertex_data, show_input_sequences, sep="_"):
    iq = 0
    outputs = []
    in_seq = []
    for s in random_samples:
        invalid = False
        iq += 1
        x_start, x_stop, y, label, map = [], [], [], [], dict()

        # get abs. parent_times for interval start
        abs_time = {}
        k_to_proc = list(s.keys())
        i = -1

        while True:
            if len(k_to_proc) == 0: break
            i += 1
            if i >= len(k_to_proc): i = 0
            k = k_to_proc[i]
            if str.startswith(k, "dL_"):
                k_to_proc.remove(k)
                continue

                # parent time is max of parents time  + my tprev
            try:
                # look for value for abs_time of all parents
                rel_time_to_me = s["dL_" + k]

                if not vertex_data[k]["parents"]:
                    pars = []
                else:
                    pars = vertex_data[k]["parents"]

                found_time = [0]
                for p in pars:
                    found_time.append(abs_time[p])

                abs_time_of_my_earliest_parent = max(found_time)  # [0]+[abs_time[p[3:]] for p in pars])

                # max([0] + [abs_time[p[3:]] for p in vertex_data["dL_" + k]["parents"] if str.startswith(p, "dL_")])


                abs_time[k] = rel_time_to_me + abs_time_of_my_earliest_parent
                k_to_proc.remove(k)
            except:
                pass  # print("Stuck") # not all parents there yet

        # plot this sample - abs_time[k] is absolute time of tprev (mean)
        overall_high = max(list(abs_time.values())) + 0.5 * max(list(abs_time.values()))  # buffer indicating end
        output_intervals = {} # key: TV value: list of tuples (value, start, end)
        for el in s:
            if str.startswith(el, "dL_"): continue

            if not sep:
                span = re.search("\d", el)
                name = el[:span.start()]
                number = int(el[span.start():])
            else:
                name = el.split(sep)[0]
                number = int(el.split(sep)[1])

            x_s = abs_time[el]
            try:
                if sep:
                    x_e = abs_time[name + sep + str(number + 1)]  #
                else:
                    x_e = abs_time[name + str(number + 1)]  #
            except:
                x_e = overall_high  # no next element

            #yy = name
            lab = s[el]
            yy =lab
            if x_e<0:
                #print("Invalid sample - negative interval time")
                invalid = True
                break
            if x_s > x_e:
                a =0
                #print("Invalid sample - Start time musst be bigger than end time")
                invalid = True
                break

            x_start.append(x_s)
            x_stop.append(x_e)
            y.append(name)
            label.append(lab)

            try:
                output_intervals[name].append([yy, x_s, x_e])
            except:
                output_intervals[name] = [[yy, x_s, x_e]]
        if not invalid:
            #print("\n")
            #print(s)
            in_seq.append(s)
            # remove nevers
            for n in output_intervals:
                occs = output_intervals[n]
                t_rem = []
                i=0
                for _ in range(len(occs)):
                    o = occs[i]

                    # if previous element same as mine I would not see it
                    pre_el_idx = i - 1
                    if pre_el_idx < 0:
                        i += 1
                        continue

                    i += 1
            outputs.append(output_intervals)

            # with nevers
            if show_input_sequences:
                print(s)
                #IntervalPlotter().plot_timeline(x_start, x_stop, y, label, y_shift=0.08, title="With Never - Sample %s" % str(iq))
                plt.show()

            # drop nevers
            i = 0
            while "Never" in label:
                idx = label.index("Never")
                # replace with previous - defined by my endinterval
                # lösche meinen Start - finde zugehörigen end
                # x_start 5   10 20
                # x_stop  10  20 40
                # val     O   N  O

                repl_idx = x_stop.index(x_start[idx])
                x_stop[repl_idx] = x_stop[idx]
                del x_start[idx]
                del x_stop[idx]
                del y[idx]
                del label[idx]

            if show_input_sequences:
                x_start = []
                x_stop = []
                y = []
                label = []
                for otto in output_intervals:
                    lst_seq = output_intervals[otto]
                    for l in lst_seq:
                        x_start += [l[1]]
                        x_stop += [l[2]]
                        y += [otto]
                        label += [l[0]]



                #IntervalPlotter().plot_timeline(x_start, x_stop, y, label, y_shift=0.08, title="Sample %s" % str(iq))
                plt.show()

    return outputs, in_seq

def sequences_to_intervals(random_samples, vertex_data, show_input_sequences, sep="_"):
    # All never occurrences are "invisible"
    # Sequence supposed to be t=20, sig="BedLicht", val='Aus_bed'   not BedLicht_1 etc. - because I do not know
    p = 0
    iq = 0
    outputs = []
    in_seq = []
    for s in random_samples:
        invalid = False
        iq += 1
        x_start, x_stop, y, label, map = [], [], [], [], dict()

        # get abs. parent_times for interval start
        abs_time = {}
        k_to_proc = list(s.keys())
        i = -1

        while True:
            if len(k_to_proc) == 0: break
            i += 1
            if i >= len(k_to_proc): i = 0
            k = k_to_proc[i]
            if str.startswith(k, "dL_"):
                k_to_proc.remove(k)
                continue

                # parent time is max of parents time  + my tprev
            try:
                # look for value for abs_time of all parents
                rel_time_to_me = s["dL_" + k]

                if not vertex_data[k]["parents"]:
                    pars = []
                else:
                    pars = vertex_data[k]["parents"]

                found_time = [0]
                for p in pars:
                    found_time.append(abs_time[p])

                abs_time_of_my_earliest_parent = max(found_time)  # [0]+[abs_time[p[3:]] for p in pars])

                # max([0] + [abs_time[p[3:]] for p in vertex_data["dL_" + k]["parents"] if str.startswith(p, "dL_")])


                abs_time[k] = rel_time_to_me + abs_time_of_my_earliest_parent
                k_to_proc.remove(k)
            except:
                pass  # print("Stuck") # not all parents there yet

        # plot this sample - abs_time[k] is absolute time of tprev (mean)
        overall_high = max(list(abs_time.values())) + 0.5 * max(list(abs_time.values()))  # buffer indicating end
        output_intervals = {} # key: TV value: list of tuples (value, start, end)
        for el in s:
            if str.startswith(el, "dL_"): continue

            if not sep:
                span = re.search("\d", el)
                name = el[:span.start()]
                number = int(el[span.start():])
            else:
                name = el.split(sep)[0]
                number = int(el.split(sep)[1])

            x_s = abs_time[el]
            try:
                if sep:
                    x_e = abs_time[name + sep + str(number + 1)]  #
                else:
                    x_e = abs_time[name + str(number + 1)]  #
            except:
                x_e = overall_high  # no next element

            #yy = name
            lab = s[el]
            yy =lab
            if x_e<0:
                #print("Invalid sample - negative interval time")
                invalid = True
                break
            if x_s > x_e:
                a =0
                #print("Invalid sample - Start time musst be bigger than end time")
                invalid = True
                break

            x_start.append(x_s)
            x_stop.append(x_e)
            y.append(name)
            label.append(lab)

            try:
                output_intervals[name].append([yy, x_s, x_e])
            except:
                output_intervals[name] = [[yy, x_s, x_e]]
        if not invalid:
            #print("\n")
            #print(s)
            in_seq.append(s)
            # remove nevers
            for n in output_intervals:
                occs = output_intervals[n]
                t_rem = []
                i=0
                for _ in range(len(occs)):
                    o = occs[i]

                    # if previous element same as mine I would not see it
                    pre_el_idx = i - 1
                    if pre_el_idx < 0:
                        i += 1
                        continue
                    if o[0] == "Never" or output_intervals[n][pre_el_idx][0] == output_intervals[n][pre_el_idx + 1][0]:
                        # 2. set times
                        # mein Startintervall wird gelöscht
                        # Vorgänger Endintervall ist mein Endintervall
                        # entferne den never wert
                        # ändere den nachfolge endwert auf den never endwert

                        output_intervals[n][pre_el_idx][-1] = output_intervals[n][pre_el_idx + 1][-1]



                        output_intervals[n].remove(o)
                    else:
                        i += 1
            outputs.append(output_intervals)

            # with nevers
            if show_input_sequences:
                print(s)
                #IntervalPlotter().plot_timeline(x_start, x_stop, y, label, y_shift=0.08, title="With Never - Sample %s" % str(iq))
                plt.show()

            # drop nevers
            i = 0
            while "Never" in label:
                idx = label.index("Never")
                # replace with previous - defined by my endinterval
                # lösche meinen Start - finde zugehörigen end
                # x_start 5   10 20
                # x_stop  10  20 40
                # val     O   N  O

                repl_idx = x_stop.index(x_start[idx])
                x_stop[repl_idx] = x_stop[idx]
                del x_start[idx]
                del x_stop[idx]
                del y[idx]
                del label[idx]

            if show_input_sequences:
                x_start = []
                x_stop = []
                y = []
                label = []
                for otto in output_intervals:
                    lst_seq = output_intervals[otto]
                    for l in lst_seq:
                        x_start += [l[1]]
                        x_stop += [l[2]]
                        y += [otto]
                        label += [l[0]]



                #IntervalPlotter().plot_timeline(x_start, x_stop, y, label, y_shift=0.08, title="Sample %s" % str(iq))
                plt.show()



    return outputs, in_seq

def put_to_final(df, max_idx):
    global ITEMSETS, FINAL_ROWS
    
    index = df["time_group"].iloc[0]
    sigs = set(eval(df["collected"].iloc[0]))
    
    
    perc = 100 * (float(index)/ float(max_idx))
    if numpy.floor(perc) % 25 == 0:
        print("Finished: " + str(int(perc)) + " % ")
    
    #FINAL_ROWS.append([indices_list, values, len_indices, len_content])
        
    for sett in ITEMSETS:

        # passt es rein
        if set(sett).issubset(sigs):
             
            if not str(sorted(list(sett))) in FINAL_ROWS:
                FINAL_ROWS[str(sorted(list(sett)))] = [[index], set(sett), 1, len(sett)]
                                
            else:
                if index not in FINAL_ROWS[str(sorted(list(sett)))][0]:
                    FINAL_ROWS[str(sorted(list(sett)))][0].append(index)
                FINAL_ROWS[str(sorted(list(sett)))][2] += 1
                    
    return df

def find_similarity_sets(segmented_df, minimal_support):
    global TRANSACTIONS, ITEMSETS, FINAL_ROWS
    TRANSACTIONS = []
    FINAL_ROWS = {}
    print("Getting going")
    segmented_df = segmented_df.groupby("time_group").apply(squash)
    
    print("FIM...")
    transactions = tuple(TRANSACTIONS)#perftesting.get_default_transactions()
    print("Got " + str(len(transactions))  + " transactions.")
    relim_input = itemmining.get_relim_input(transactions)
    ITEMSETS = itemmining.relim(relim_input, min_support=minimal_support)
    
    # clean for closed frequent patterns
    itemsets1 = []
    print("Closing the patterns...")
    for s in ITEMSETS:
        can_add = True
        for j in range(len(itemsets1)):            
            if set(s).issubset(itemsets1[j]):
                can_add = False
                break            
            if set(s).issuperset(itemsets1[j]):
                itemsets1[j] = set(s)
                can_add = False
                break                
        if can_add: 
            itemsets1.append(s)            
    ITEMSETS = itemsets1
                
    
    # per itemset determine rows
    print("Per Window go...")
    segmented_df.index = range(len(segmented_df))
    max_idx = segmented_df["time_group"].max()
    segmented_df = segmented_df.groupby("time_group").apply(lambda x: put_to_final(x,max_idx))
    
    # write result
    res_df = pandas.DataFrame([[str(a) for a in r[:-2]]+[r[-2], r[-1]] for r in list(FINAL_ROWS.values())], columns=["indices", "values", "length_indices", "length_content"])
    res_df = res_df[res_df["length_content"] >= minimal_support]
      
    return res_df.groupby("indices").apply(lambda x: x.iloc[-1])
    

def invertDictionary(orig_dict):
    return {v : k for k, v in orig_dict.items()}
    

def create_state_vector(df):
    col_names = df.columns.tolist()
    
    delete_rows = ['timestamp', 'validity_invalid_items', 'session_id', 'data_type', 'outlier', 'signal_name_short', 'Signalname']
    for el in delete_rows:
        try:
            col_names.remove(el)
        except:
            pass
    col_names.sort()
    
    df["state_vector"] = ""
    first = True
    for col_name in col_names:
        if first:
            first = False
        else:
            df["state_vector"] += ";"
        df["state_vector"] += col_name
        df["state_vector"] += "="
        df["state_vector"] += df[col_name].astype('str')
    return df

def _new_column(df):
    #global __DONE_NUMBER, __NEW_COL_THRSHLD
    if df["diff"] > Keep.NEW_COL_THRSHLD:  
        print("Found time window: "+str(Keep.DONE_NUMBER))
        Keep.DONE_NUMBER +=1 
    df["time_group"] = str(Keep.DONE_NUMBER)
    return df

def _new_column_and_state_vec(df):
    #global __DONE_NUMBER, __NEW_COL_THRSHLD
    Keep.MISTER_LOCK.acquire()
    if df["diff"] > Keep.NEW_COL_THRSHLD:  
        print("Found time window: "+str(Keep.DONE_NUMBER))
        Keep.DONE_NUMBER +=1         
    df["time_group"] = str(Keep.DONE_NUMBER)
    Keep.MISTER_LOCK.release()
    
    df["value"] = df["Signalname"] + "=" + str(df["interpreted_value"])
    return df
    
def _new_column_and_state_vec_apply(data):
    ''' parallel execution for split '''
    data = data.apply(_new_column_and_state_vec, axis = 1)
    return data    
    
def parallelize_dataframe(df, func, num_cores = 40, num_partitions = 20):
    ''' used to execute the python apply command in parallel '''
    df_split = numpy.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df
        

def state_vec(line):
    '''
    Aus a=1;b=3;c=5 mache mit auswahl Signalname z.B. b -> b=3
    wobei b in Spalte Signalname und a=1,... in Spalte state_vector steht
    '''

    line["value"] = line["Signalname"] + "=" + str(line[line["Signalname"]])#line["Signalname"] + line["state_vector"].split(line["Signalname"])[1].split(";")[0]
    return line

def state_vec_new(line):
    '''
    Aus a=1;b=3;c=5 mache mit auswahl Signalname z.B. b -> b=3
    wobei b in Spalte Signalname und a=1,... in Spalte state_vector steht
    '''

    line["value"] = line["Signalname"] + "=" + str(line["interpreted_value"])#line["Signalname"] + line["state_vector"].split(line["Signalname"])[1].split(";")[0]
    return line

    
def segmentation_by_time(df, threshold_ns = 3000000000, start_col = "timestamp", end_col = "end_timestamp"):
    '''
    Where difference between end_col and start_col is bigger than threshold a seperation
    of time windows is performed and each window is assigned a number    
    '''
    #global __DONE_NUMBER, __NEW_COL_THRSHLD
    Keep.NEW_COL_THRSHLD = threshold_ns # in nanoseconds
    Keep.DONE_NUMBER = 0
    
    df[end_col] = df[start_col].shift(-1)
    df["diff"] = df[end_col] - df[start_col]
    
    df = df.apply(_new_column, axis = 1)
    #df = parallelize_dataframe(df, _new_column_and_state_vec_apply)
    return df
    


    
def fuse(a):
    ''' Merge elements with same column value e.g. 
        a = data.groupby("timestamp")
        data = a.apply(fuse)'''
    
    return a.iloc[-1:]

def join_by_columns(df_1, df_2, join_col_1, join_col_2):
    return pd.merge(df_1, df_2,  how='left', left_on=[join_col_1, join_col_2], right_on = [join_col_1, join_col_2])
    


def aggregate_sequence(df):
    '''
    Aggregates sequence by index
    in value steht a=1 in Zeile 1, b=1 in Zeile 2... und time_group z.B. 20 sagt nach was zusammengefasst wird
    daraus mache ueber groupby und apply pro Gruppe -> [20, [(ts1, a=1), (ts2, b=1), ...]
    '''
    
    resulting = []
    for i in range(len(df)):
        resulting.append((df.iloc[i]["timestamp"],df.iloc[i]["value"]))
    resulting.sort(key=lambda x: x[0])    
    df["sequence_wo_timefuse"] =  str(resulting)
    df = df.head(1)
    df["sequence_index"] = df["time_group"]
    return df
    
def select_from_list(df, column, index_list, indices_support, value_support, idx):
    '''
    extracts all elements in df and column column
    selects a list of indices e.g. [293, 295, 395, 496, 305, 377]
    '''
    if isinstance(index_list, list):
        valid_indices = index_list
    else:
        valid_indices = eval(index_list)
    
    df1 = df[df[column].isin(valid_indices)]
    #df2 = df1.apply(state_vec, axis=1)
    df_out = df1.groupby(column).apply(aggregate_sequence)
    
    df_out = df_out[['sequence_index', 'sequence_wo_timefuse']]
    df_out["set_index"] = idx
    df_out["length_indices"] = indices_support
    df_out["length_content"] = value_support
    
    return df_out

def select_all_from_list(df_indices, df_original, ignore_idx = []):
    df_indices["aggregated"] = "FILL"
    df_original["time_group"] = pd.to_numeric(df_original["time_group"])
    
    # Parallel Version
    first = True
    res_df = None
    
    inputs = []
    for idx in range(len(df_indices)):
        if idx in ignore_idx: continue
        inputs.append([df_original, "time_group", df_indices.iloc[idx]["indices"], df_indices.iloc[idx]["length_indices"], df_indices.iloc[idx]["length_content"], idx])
    res_list = parallelize_stuff(inputs, select_from_list)
    
    for res in res_list:
        if first:
            res_df = res; first = False
        else:
            res_df = res_df.append(res)

    '''

    # Standard version    
    start = time.clock()
    first = True
    res_df = None
    for idx in range(len(df_indices)):
        if idx in ignore_idx: continue
        print("Processing Index: " + str(idx+1) + " | "+ str(len(df_indices)))        
        calc = select_from_list(df_original, "time_group", df_indices.iloc[idx]["indices"], df_indices.iloc[idx]["length_indices"], df_indices.iloc[idx]["length_content"], idx)
        calc["set_index"] = idx
        calc["length_indices"] = df_indices.iloc[idx]["length_indices"]
        calc["length_content"] = df_indices.iloc[idx]["length_content"]
        
        if first:
            res_df = calc; first = False
        else:
            res_df = res_df.append(calc)
        df_indices.ix[idx, "aggregated"]= str(select_from_list(df_original, "time_group", df_indices.iloc[idx]["indices"], df_indices.iloc[idx]["length_indices"], df_indices.iloc[idx]["length_content"], idx))
    
    print ("Sequential Version: " + str(time.clock() - start))
    #print ("Result DF: " + str(len(res_df)))
    ''' 
    return res_df.reset_index()[["sequence_index", "sequence_wo_timefuse", "set_index", "length_indices", "length_content"]]

def parallelize_stuff(list_input, method, simultaneous_processes = 10, print_all =False):
    if len(list_input) <= simultaneous_processes:
        simultaneous_processes = len(list_input)
    '''
    The smarter way to loop - 
    list_input is a list of list of input arguments [[job1_arg1, job1_arg2, ...], [job2_arg1, job2_arg2, ...], [job3_arg1, job3_arg2, ...]] 
    and method is the method to compute from the input arguments
    the result is a list of output arguments from the given method [jobK_res, jobL_res, ...]

    Here: Lose order of in to output
    '''
    # Initialize
    process_number = len(list_input)
    split_number = simultaneous_processes # split by groups of 10
    task_queue = Queue()
    done_queue = Queue()
    
    cur_runs = 0

    # Submit tasks jedes Put hat: (methode, argumente_tuple) z.B. (multiply, (i, 7))
    for list_in in list_input:
        task_queue.put((method,  list_in))
    
    # Start worker processes
    jobs =[]        
    # Split tasks by defined number
    for i in range(process_number):
        if print_all: print("Starting task "+str(i+1))
        p = Process(target=_worker, args=(task_queue, done_queue))
        jobs.append(p)

    # Get and print results
    output_list = []
    j = 0
    for i in range(len(jobs)):
        if cur_runs < split_number:
            if print_all: print("Start job: "+str(i+1))
            jobs[i].start()
            cur_runs +=1
            if len(jobs) != split_number and (len(jobs) - i - 1) < split_number:# remaining_jobs = len(jobs) - i - 1
                j += 1
                if print_all: print("Received results "+str(j+1) + " | " + str(1+len(list_input)))
                output_list.append(done_queue.get())
                #print("Got: "+ str(done_queue.get().head(1)))
            if len(jobs) == split_number and (i +1) == split_number:# remaining_jobs = len(jobs) - i - 1
                j += 1
                if print_all: print("Received results "+str(j+1) + " | " + str(1+len(list_input)))
                output_list.append(done_queue.get())
                #print("Got: "+ str(done_queue.get().head(1)))     
            
        else:
            j += 1
            if print_all: print("Received results "+str(j+1) + " | " + str(1+len(list_input)))
            output_list.append(done_queue.get())


    while j != len(list_input):        
        res = done_queue.get()                
        j += 1
        if print_all: print("Received results "+str(j+1) + " | " + str(1+len(list_input)))
        output_list.append(res)
        

    # End all 
    for i in range(process_number):
        task_queue.put('STOP')
    
    for job in jobs:
        try:
            job.shutdown()
        except: 
            pass
    return output_list

def all_aggregated_info_to_sequence(df):   
    '''
    from a database with sequence sets extract lists of sequences that
    can be processed
    '''
    
    # parallelize
    max_index = df["set_index"].max() 
    input_args = []
    for i in range(max_index):
        input_args.append([df, i])
    
    # jobs
    output_list = parallelize_stuff(input_args, aggregated_info_to_sequence)
    
    # resulting df
    first = True
    for res in output_list:
        if first:
            res_df = res; first = False
        else:
            res_df = res_df.append(res)
    return res_df

def sort_and_fuse_ts(line, col, res_col, remove_ts):
    '''
    for input list of type [(ts1, val), (ts2, val),...] which is in col of dataframe line
    fuse the list (i.e. items with same timestamp are grouped and sort by ts
    
    if remove_ts: False
        results in [{ts1:[val1, val2, ...]: (ts, ...}
    if remove_ts: True
        results in item set [[val1, val2, ...], [valx, valy, ...], ...]
    
    which is stored in column res_col
    '''
    lst = eval(line[col])
    known_ts = {}
    for el in lst:
        if el[0] in known_ts:
            known_ts[el[0]].append(el[1])
        else:
            known_ts[el[0]] = [el[1]]
    if not remove_ts:
        line[res_col] = collections.OrderedDict(sorted(known_ts.items()))
    else:
        line[res_col] = [a[1] for a in sorted(known_ts.items())]
    return line

def aggregated_info_to_sequence(data_frame, set_idx):
    '''        
    if they happen in identical timeslots they are grouped!
    Result is a sequence of itemsets [[a,n]...  
    
    '''
    cur_df = data_frame[data_frame["set_index"]==set_idx]
    
    # group and sort sequences by timestamp
    cur_df = cur_df.apply(sort_and_fuse_ts, args = ("sequence_wo_timefuse", "sequence", True), axis=1)
    
    return cur_df
    
def symbolize_values(df_set, to_number):
    '''
    In order to be processed more easy, strings will be named with short consecutive letters
    i.e. a unique shortname will be assigned to each unique string
    '''
    global _KNOWN_DICT, __KNOWN_DICT_CNT
    df_set["translator"] = str(_KNOWN_DICT)
    
    k = -1
    for sequence in df_set["sequence"].tolist():        
        k+=1
        for itemset in sequence: 
            for i in range(len(itemset)):
                if itemset[i] not in _KNOWN_DICT:
                    if to_number: _KNOWN_DICT[itemset[i]] = str(__KNOWN_DICT_CNT); __KNOWN_DICT_CNT += 1 
                    else: _KNOWN_DICT[itemset[i]] = "gen"+str(__KNOWN_DICT_CNT); __KNOWN_DICT_CNT += 1            
                itemset[i] = _KNOWN_DICT[itemset[i]]
    return df_set
    
def all_symbolize_values(df):
    global _KNOWN_DICT, __KNOWN_DICT_CNT
    _KNOWN_DICT = {}
    __KNOWN_DICT_CNT = 0
    
    start = time.clock()   
    df1 = deepcopy(df).groupby("set_index").apply(lambda x: symbolize_values(x, True))
    print ("Time for symbolization: " + str(time.clock() - start))
    df1["translator"] = str(_KNOWN_DICT)
    
    return df1
    
def desymbolize_sequential_pattern(level_dict, mapping_dict):
    mapping_dict = invertDictionary(mapping_dict)
    
    for el in level_dict:
        for i in range(len(level_dict[el])):
            for j in range(len(level_dict[el][i][0])):
                level_dict[el][i][0][j] = [mapping_dict[k1] for k1 in level_dict[el][i][0][j]]
            print("\nFound patterns: \n"+"\n".join(["|".join(o) for o in level_dict[el][i][0]]))
    return level_dict
    
def find_n_grams(n, single_sequence):
    ''' 
    Find n grams in Sequence, n=2 means pattern of length two, n=3 of length 3, ...
    n = pattern length
    e.g. in 
    3 6 1 2 7 3 8 9 7 2 2 0 2 7 2 8 4 8 9 7 2 4 1 0 3 2 7 2 0 3 8 9 7 2 0
    find
    3 6 1 [2 7] 3 [8 9 7 2] 2 0 [2 7] 2 8 4 [8 9 7 2] 4 1 0 3 [2 7] 2 0 3 [8 9 7 2] 0
    '''    
    
    grams = [single_sequence[i:i+n] for i in xrange(len(single_sequence)-n)]
    
    return itemfreq(grams)

def mine_java_spmf(jar_path, sequences, algorithm, *args):


    res = run_java_spmf(jar_path, sequences, {}, algorithm, *args)

    return res

def all_run_java_spmf(jar_path, sequence_df, algorithm, *args):

    
    # parallel - slower?
    start = time.clock() 
    # Job Input
    input_lst = [] 
    for set_index in list(sequence_df["set_index"].drop_duplicates()):
        input_lst.append([set_index, jar_path, sequence_df, deepcopy(_KNOWN_DICT), algorithm, *args])

        # debug:
        algorithm = "PrefixSpan"
        res = run_java_spmf_parallel(set_index, jar_path, sequence_data, {}, algorithm, *args)

    # Job execution
    output_lst = parallelize_stuff(input_lst, run_java_spmf_parallel, simultaneous_processes=2 )
    
    # job result
    res_df = _append_df_list(output_lst)        
        
    print ("Time for Java: " + str(time.clock() - start))
    return res_df

def _append_df_list(output_lst):
    first = True
    for res in output_lst:
        if first:
            res_df = res; first = False
        else:
            res_df = res_df.append(res)
    return res_df


def print_result_pattern_spmf(df, column, min_level = 0):
    i = 0
    for pattern in list(df[column]):
        i+=1
        cur_dict = eval(pattern)
        print("\n\n-------- Sequence " + str(i))
        for k in cur_dict.keys():
            if k < min_level: continue
            print("\nLevel "+str(k))
            print("\n\n".join(["Sup: " + str(a[1]) + " - Pattern" + str(a[0])  for a in cur_dict[k]]))
            #print("\n\nPattern: ".join([str(",".join(a[0])) for a in cur_dict[k]]))      


def run_java_spmf(jarPath, sequence_sets, reverse_dict, algorithm, *args):
    '''
    ALL Sequences must be numbers
    Input format: [[1,2,3], [2,3]]
    algorithm: PrefixSpan   with args: 50%
    '''
    # ASSUME SEQUENCE SET ALREADY IN RIGHT FORMAT ONLY NUMBERS!
    set_index = -1

    # 0. temporary path
    import numpy as np
    tmp_path =r"C:\temporary\inputSequences"+str(np.random.rand()) + ".txt"

    java_jre8_path = r'C:\Program Files (x86)\JavaSoft\jre\1.8.0_121\bin\java'

    # 1. map to numbers
    stored_map = invertDictionary(reverse_dict)

    # 2. Write temporary file
    # TEST


    open(tmp_path, 'w').close()
    with open(tmp_path, 'a') as the_file:
        first = True
        for sets in sequence_sets:
            #sets = sets[:10]   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ACHTUNG
            if first:
                the_file.write(""+" -1 ".join([str(a) for a in sets])+" -2")
                first = False
            else:
                the_file.write("\n"+" -1 ".join([str(a) for a in sets])+" -2")

    # 3. Run Java on this
    from subprocess import Popen, PIPE, STDOUT
    #print("Doin da Popen: %s" %str([[java_jre8_path, '-jar', jarPath, "run", algorithm, tmp_path, tmp_path+".done"]+ list(args)]))
    #p = Popen([java_jre8_path, '-jar', jarPath, "run", algorithm, tmp_path]+ list(args), stdout=PIPE, stderr=STDOUT)
    p = Popen([java_jre8_path, '-jar', jarPath, "run", algorithm, tmp_path, tmp_path+".done"] + list(args), stdout=PIPE, stderr=STDOUT)
    p.communicate()
    # 4. read result as dictionary
    eval_dict = {}
    result_dict = {}
    result_dict_un = {}


    # read
    out = tmp_path+".done"
    while not os.path.exists(out):
        time.sleep(0.1)
    #output: ((sup1, pat1), (sup2, pat2))
    with open(out) as f:
        all_patterns = []
        for line in f.readlines():
            if not line:continue
            input = line.replace("\n", "")
            try:
                support = int(input.split(": ")[-1].replace(" ", ""))
            except:
                continue
            iter = line.split(" #SUP")[0]
            pattern = [int(e.replace(" ", "")) for e in iter.split("-1") if e != " " and e]
            all_patterns += [(support, pattern)]
        f.close()
    os.remove(tmp_path)



    return all_patterns

def run_java_spmf_parallel(set_index, jarPath, sequence_df, reverse_dict, algorithm, *args):
    '''
    runs the SPMF EntryPoint and reads out results    
    '''
    # ASSUME SEQUENCE SET ALREADY IN RIGHT FORMAT ONLY NUMBERS!
    sequence_sets = sequence_df[sequence_df["set_index"] == set_index]["sequence"].tolist()

    # 0. temporary path
    tmp_path =r"C:\temporary\inputSequences"+str(set_index) + ".txt"
    
    java_jre8_path = r'C:\Program Files (x86)\JavaSoft\jre\1.8.0_121\bin\java'
    
    # 1. map to numbers
    stored_map = invertDictionary(reverse_dict)

    # 2. Write temporary file
    # TEST


    open(tmp_path, 'w').close()
    with open(tmp_path, 'a') as the_file:    
        first = True
        for sets in sequence_sets:      
            #sets = sets[:10]   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ACHTUNG 
            if first: 
                the_file.write(""+" -1 ".join([" ".join(a) for a in sets])+" -2")
                first = False
            else: 
                the_file.write("\n"+" -1 ".join([" ".join(a) for a in sets])+" -2")
    
    # 3. Run Java on this        
    from subprocess import Popen, PIPE, STDOUT
    print("Doin da Popen: %s" %str([[java_jre8_path, '-jar', jarPath, algorithm, tmp_path]+ list(args), PIPE, STDOUT]))
    p = Popen([java_jre8_path, '-jar', jarPath, algorithm, tmp_path]+ list(args), stdout=PIPE, stderr=STDOUT)

    # 4. read result as dictionary
    # dict[level] = [pattern_list, support]
    # lese auch eval raus: dict["pattern_number"] und dict[total_time] und dict["memory"]
    eval_dict = {}
    result_dict = {}
    result_dict_un = {}

    for line in p.stdout:
        #print(line)
        if line[:11] == b" Total time": eval_dict["total_time"] = str(line[12:]).replace("\\n", "").replace("b'", "").replace("'", "")
        if line[:28] == b" Frequent sequences count : ": eval_dict["patterns_found"] = str(line[28:]).replace("\\n", "").replace("b'", "").replace("'", "")
        if line[:17] == b" Max memory (mb):": eval_dict["max_memory_mb"] = str(line[17:]).replace("\\n", "").replace("b'", "").replace("'", "")
        
        if line[:5]==b"Level": 
            input = [el.split("#") for el in str(line[8:]).replace("\\n", "").replace("b'", "").replace("'", "").split(";")][:-1]
            all_patterns = []
            all_patterns_un = []
            for lev in input: 
                support = int(lev[1].replace("SUP: ",""))
                patterns_raw = [e.lstrip()[::-1].lstrip()[::-1].split(" ") for e in lev[0].split("-1") if e != " "]
                patterns_un = []
                patterns = []
                for p in patterns_raw:
                    patterns.append([stored_map[pi.replace(":", "")] for pi in p])
                    patterns_un.append([pi.replace(":", "")  for pi in p])
                all_patterns.append([patterns, float(support)/float(len(sequence_sets))])
                all_patterns_un.append([patterns_un, float(support)/float(len(sequence_sets))])
            result_dict[int(str(line[:7])[-2])] = all_patterns
            result_dict_un[int(str(line[:7])[-2])] = all_patterns_un
    os.remove(tmp_path)
    
    #sequence_df["all_sequences"] = str(sequence_df["sequence"].tolist())
    sequence_df["set_index"] = set_index
    sequence_df["sequence"] = str(sequence_sets)
    
    sequence_df["pattern_found"] = str(result_dict)
    sequence_df["pattern_found_un"] = str(result_dict_un)
    
    try:
        sequence_df["eval_total_time"] = eval_dict["total_time"]
    except:
        sequence_df["eval_total_time"] = "0"
    try:
        sequence_df["patterns_found"] = eval_dict["patterns_found"]
    except:
        sequence_df["patterns_found"] = 0
    try:
        sequence_df["max_memory_mb"] = eval_dict["max_memory_mb"]
    except: 
        sequence_df["max_memory_mb"] = 0
    return sequence_df.head(1)


# ------------- parallelization stuff ------------- 
  
    
def _worker(input, output): # Function run by worker processes
    for func, args in iter(input.get, 'STOP'):
        result = _calculate(func, args)
        output.put(result)
        
def _calculate(func, args): # Function used to calculate result
    result = func(*args)
    return result
       
       
       
       
