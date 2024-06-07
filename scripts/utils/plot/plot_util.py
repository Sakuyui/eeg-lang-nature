import numpy as np
from functools import reduce
import matplotlib.pyplot as plt

def generate_random_color_rgb_descriptions(color_count):
    color_set = set()
    def random_generate_one_color():
        return tuple(np.random.randint(0, 255, color_count))
    def get_hex(number):
        return f'{number:0>2X}'
    while len(color_set) < color_count:
        color_set.add(random_generate_one_color())
    return ['#' + reduce(lambda x, y: x + y, [get_hex(color[i]) for i in range(3)]) for color in color_set]

def check_and_fix_time_range(points, sel_times_from_slot_id = -1, sel_times_to_slot_id = -1):
    if sel_times_from_slot_id < 0 or sel_times_from_slot_id >= max(points):
        sel_times_from_slot_id = 0
    if sel_times_to_slot_id < 0 or sel_times_to_slot_id >= max(points):
        sel_times_to_slot_id = max(points)
    if sel_times_from_slot_id > sel_times_to_slot_id:
        sel_times_from_slot_id = 0
        sel_times_to_slot_id = max(points)
    return (sel_times_from_slot_id, sel_times_to_slot_id)

def plot_segments(end_points, linewidth = 2, segment_categories = None, colors = None, sel_times_from_slot_id = -1, sel_times_to_slot_id = -1):
    sel_times_from_slot_id, sel_times_to_slot_id = check_and_fix_time_range(end_points, sel_times_from_slot_id, sel_times_to_slot_id)
        
    current_time_point_index = 1
    cnt_end_points = len(end_points)
    sorted(end_points)

    while current_time_point_index < cnt_end_points:
        time_point = end_points[current_time_point_index]
        if time_point > sel_times_to_slot_id:
            break
        if time_point >= sel_times_from_slot_id:
            width = end_points[current_time_point_index] - end_points[current_time_point_index - 1]
            if end_points[current_time_point_index - 1] - sel_times_from_slot_id >= 0:
                plt.axvspan(end_points[current_time_point_index - 1] - sel_times_from_slot_id, \
                            end_points[current_time_point_index] - sel_times_from_slot_id, \
                            facecolor='#aabbcc' if segment_categories is None or colors is None else colors[segment_categories[current_time_point_index - 1]],
                            alpha=0.1)
                # print("draw retangle from %d to %d" % (end_points[current_time_point_index - 1],\
                #                                       end_points[current_time_point_index]))

        current_time_point_index += 1

def plot_segment_lines(time_points=[], linewidth = 2, sel_times_from_slot_id = -1, sel_times_to_slot_id = -1):
    sorted(time_points)
    sel_times_from_slot_id, sel_times_to_slot_id = check_and_fix_time_range(time_points, sel_times_from_slot_id, sel_times_to_slot_id)

    # TODO: optimize the linear search. Previously make sequential sorted.
    print("segment points = %s, plot from time slot %d to time slot %d" % (time_points,  sel_times_from_slot_id, sel_times_to_slot_id))
    for time_point in time_points:
        if time_point >= sel_times_from_slot_id and time_point <= sel_times_to_slot_id:
            plt.axvline(x=time_point - sel_times_from_slot_id, ymin=0, ymax = 1.0, linewidth=2, color='blue')
            plt.text(time_point - sel_times_from_slot_id, 0, str(time_point), fontsize=45, color="blue")

def seconds_to_time_slots(times_in_second, sampling_rate_hz):
    time_slots = [time_sec * sampling_rate_hz for time_sec in times_in_second]
    return time_slots

def plot_signal_multiple_channels(signal_data, channel_names, sampling_rate_hz, time_slot_from = -1, time_slot_to = -1):
    cnt_total_time_slots = signal_data.shape[1]
    if(time_slot_from < 0 or time_slot_from >= cnt_total_time_slots):
        time_slot_from = 0
    if(time_slot_to < 0 or time_slot_to >= cnt_total_time_slots):
        time_slot_to = cnt_total_time_slots
        
    print("plot from %lf sec to %lf sec" % (time_slot_from / sampling_rate_hz, time_slot_to / sampling_rate_hz))
    focusing_signal_data = signal_data[:, time_slot_from: time_slot_to]
    y_scale = np.max(focusing_signal_data) - np.min(focusing_signal_data)
    cnt_time_slots = (time_slot_to - time_slot_from) # * sampling_rate_hz
    print("time slots count: %d" % cnt_time_slots)
    time_point_sample_rate = 0.2
    x = np.linspace(0, \
                    cnt_time_slots, \
                    (int)(cnt_time_slots * time_point_sample_rate), \
                    dtype=np.int32, \
                    endpoint=False)
    fig, ax = plt.subplots(figsize = (175, 135))
    
    cnt_channels = len(signal_data.shape[0])
    for i in range(cnt_channels):
        ax.plot(x, np.take(focusing_signal_data, x, axis = 1)[i, :] + y_scale * i, linewidth=3.0)
        
    labels = channel_names
    ax.set_yticks(np.arange(0, cnt_channels) * y_scale)
    ax.set_yticklabels(labels, fontsize = 45)
    x_ticks = np.arange(0, time_slot_to - time_slot_from, (time_slot_to - time_slot_from) * (time_point_sample_rate), dtype=np.int32)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(labels = [str(x + time_slot_from) for x in x_ticks], fontsize = 45)

    del focusing_signal_data
    