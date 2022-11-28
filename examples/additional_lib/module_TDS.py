#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for Signal processing labs

Centrale Lille

Created on Dec. 12, 2017
    
@author: Pierre Chainais
"""

from numpy import *
from matplotlib.pyplot import *
from numpy.random import randn

# To record and listen to sounds/signals
#import sounddevice as 


# For interactive widgets
from ipywidgets import interact
from bokeh.io import push_notebook, output_notebook
from bokeh.io import show as bkshow
from bokeh.plotting import figure as bkfigure
output_notebook()

from scipy.spatial import KDTree
import scipy.signal as sg

# With the new version of ipywidgets, I ran into this problem, but then I found on their github readme that you now need to follow
#
# pip install ipywidgets
#
# with
#
# jupyter nbextension enable --py widgetsnbextension
#
# That cleared the problem up for me.

# For Bokeh :
# jupyter notebook --NotebookApp.iopub_data_rate_limit=100000000


def display_signal(x,t=array([0]),titleroot ='Signal no '):
    """To display a signal x or a set of signals x[i,:] within array x
    as a function of time t with generic title of the form titleroot i """

    if t.size<2:
        t=arange(max(x.shape))

    # Test de saturation du signal
    sature=False
    if len(abs(x)>=0.999)>5:
        sature=True
        # print("Attention ! Le signal sature...")

    # Affichage
    # N = x.shape(0)   # nombre de signaux acquis

    if len(x.shape)<2:
        nb_sig=1
    else:
        nb_sig = max(1,min(x.shape))

    if nb_sig<2:
        fig = plot(t,x,linewidth=2);
        title(titleroot)
        grid(True)
        gcf().set_figheight(6)
        gcf().set_figwidth(10)
        show()
        return fig, sature
    else:
        # print(nb_sig)
        fig, ax = subplots((2+nb_sig)//3,3,sharex=True);
        for i, axf in enumerate(ax.flatten()[0:nb_sig]):
            axf.plot(t,x[i,:],linewidth=2);
            axf.set_title(titleroot + str(i))
            axf.grid(True)
        f=gcf();
        f.set_figheight(5*((2+nb_sig)//3))
        f.set_figwidth(14)
        fig.tight_layout()
        show()
        return fig, sature


# REVOIR CHOIX POSITION DES DETECTIONS ?
def segment_signal(x, N=3, seuil=1, Fe=22050, duration=0.05):
    """To segment signal x into N parts corresponding to N impulse responses.
    x : signal to segment
    seuil : threshold for detection of peaks
    N : number of patterns to detect and segment
    duration : assumed minimal duration of an individual pattern [in seconds].
    """

    if max(abs(x))<1:
        seuil=seuil*max(abs(x))     # pour définir un seuil relativement au maximum du signal
        if max(abs(x))<1e-10:
            print('\033[1;31;43m Your signal is to weak... Please check it has non zero values.')
            # voir http://ozzmaker.com/add-colour-to-text-in-python/

            #raise ValueError('Your signal is to weak... Please check it has non zero values.')
            #sys.stderr.write('toto')
            return

    # Idée : détecter le 1er pic > seuil, puis le suivant au moins espacé de min_duration, etc...
    # puis faire descendre le seuil si pas assez de signaux détectés jusqu'à détextion de N motifs.

    detec=array([])

    while (detec.size<N) & (seuil>0.05) :    # à la recherche du plus petit seuil assurant N détections
        next_pos=0

        #detec_list= find(abs(x)>seuil)
        detec_list= where(abs(x)>seuil)[0]

        n=0

        if size(detec_list)>0:
            detec = detec_list[0]
            n+=1
            next_pos = detec_list[0] + int(duration*Fe)   # on interdit des détections trop rapprochées

            while (size(detec_list[detec_list>next_pos])>0) & (next_pos<x.size) & (n<N) & (detec.size<N):
                n+=1
                ind = detec_list[detec_list>next_pos]
                #print(len(detec))
                # nb_detec = nb_detec+1 #ind_pos = int(detec[0] + min_duration*Fe)
                detec = hstack((detec, ind[0]))   # 1er prochain passage de seuil en valeur absolue
                next_pos = ind[0] + int(duration*Fe)   # on interdit des détections trop rapprochées

        seuil = 0.9*seuil



    # detec_list = next_pos + where(abs(x[next_pos:])>seuil)
    # print(detec.size)

    prop = 0.01

    if detec.size>0:
        if detec.size==1:
            y = x[detec-int(duration*Fe*prop):detec+int(duration*Fe)].T
        else:
            y = zeros((detec.size,int(duration*Fe*prop)+int(duration*Fe)))
            for n in range(detec.size):
                y[n,:] = x[detec[n]-int(duration*Fe*prop):detec[n]+int(duration*Fe)].T
    return y,detec



def matched_filter(x, y, init_threshold = 0.5, interact = True, lagmin = 256):
    """Computes the output of the matched filter to detect the presence of pattern y within signal x.
    Inputs :
        x : signal
        y : pattern to be detected
    Returns :
        z :
        h : impulse response of the matched filter (fliplr(y) indeed)

    See function threshold_detection for later detection at a variable
    threshold starting from init_threshold.
    """
    # Matched filtering
    h = y[::-1]  # réponse impulsionnelle du filtre adapté au motif y
    z = convolve(x, h,'valid');  # convolution => filtrage adapté de x par h.
    # ou plutôt xcorr pour normaliser ?  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< REVOIR
    # z = xcorr(x,y)

    return z, h


def threshold_detection(z, Fs=2205, threshold = 0.5, lagmin=100):
    """Detection by thresholding signal z (e.g. from matched filter) at a variable threshold
    starting from init_threshold. This function is interactive (slider to choose the threshold)
    if interact=True (default)
    Inputs :
        z : the signal to be thresholded
        Fs : sampling frequency, default is Fs = 22050
        init_threshold : the default threshold (should be between 0 and 1)
        lagmin : minimal time lag between 2 successive detection instants (to avoid multiple detections)
    """

    Nz = z.size

    overthresh = where(z>threshold)[0]
    if size(overthresh)>0:
        k1 = overthresh[0]   # first time the threshold is crossed
    else:
        d, val = -1, 0
        return d, val
    #print(size(z[k1:min((Nz,k1+lagmin))]))


    if k1.size>0:   # looking for next point that is after decreasing and at least lagmin away
        k2 = k1 + lagmin + where(z[k1+lagmin:]<0.9*max(z[k1:min((Nz,k1+lagmin))]))[0][0]
        n = 1
    else:
        n = 0

    d = array([])
    val = array([])

    while (k1.size>0) & (k2.size>0) & (k2-k1>0):
        #print(k1)
        #print(k2)
        tmpval, pos = max(z[k1:k2]), argmax(z[k1:k2]) # detecting the maximum of response within k1:k2

        d = hstack((d, k1+pos))      # stacking new detection position and value
        val = hstack((val, tmpval))

        if (k2+lagmin<Nz) & (where(z[k2+lagmin:]>=threshold)[0].size>0):   # checking whether the end of signal has been reached
            k1 = k2 + lagmin + where(z[k2+lagmin:]>=threshold)[0][0]  # searching for next detection
        else:
            k1=array([])

        if (k1.size>0) & (k1+lagmin<Nz):
            #print(size(z[k1:min((Nz,k1+lagmin))]))
            k2 = k1 + lagmin + where(z[k1+lagmin:]<0.9*max(z[k1:min((Nz,k1+lagmin))]))[0][0]
        #else:
        #    k2=[]

        n=n+1

        d = [int(i) for i in d]   # to get a list of integers

    return d, val


def filtre_detection(events, lagmin=256):
    """To avoid multiple detections within a short time interval
    determined by lagmin. The algorithm is recursive to keep only
    one among 2 close events. The output is a filtered list of events
    and corresponding values where the event with the highest value is
    kept. The 2 first attributes of an event must be a list of indices
    and an array of values : d, val."""

    index = events[0]
    value = events[1]
    filt_events = events

    nb_events = index.len

    if nb_events<2:
        return filt_events
    else:
        dist_events = abs(diff(index)) # BE CAREFUL index.len>1 !!!

        if min(dist_events)>lagmin:  # stopping criterion
            return index, value
        else:                        # recursion
            imin = argmin(dist_events)
            print(imin)
            ind_to_del = argmin([value[imin], value[imin+1]])
            index.pop(ind_to_del)
            print(index[ind_to_del])
            value = hstack((value[0:imin+ind_to_del], value[imin+ind_to_del+1:]))
            filt_events = (index, value)
            filtre_detection(filt_events, lagmin)


def evaluate_detection(events, true_events, margin = 1024):
    """Takes events and true_events as inputs.
    events[0] = indices of events while events[1] are the corresponding values.
    true_events = indeices of reference events = array.
    margin = minimum interval between two detected events
    lagmin = margin//2 = interval within which events are considered as matching.
    Outputs:
        TP, FN, FP
        TP_index = indices of true positive detections
    Usage :
        TP, FN, FP, TP_pos, TP_ind = evaluate_detection(events, true_events, margin=512)
    """

    lagmin = margin//2

    if type(events[0]) is int:
        index = events[0]
    else:
        index = array(events[0]).reshape(-1,1)
    
    value = events[1]

    index_ref = true_events.reshape(-1,1)
    #value_ref = true_events[1]

    # TP and FN w.r.t true events   <<<<<<<<<<
    # TP and FN w.r.t true events
    # Default values if index is empty (no detection)
    TP = 0
    FP = 0
    FN = index_ref.size
    check_ref=(array([]),array([]))

    if index_ref.size>0:
        t_ref = KDTree(index_ref)
        if type(index) is not int:
            check_ref = t_ref.query(index)

    ref_valid = where(check_ref[0]<=lagmin)[0]
    ind_ref = check_ref[1][ref_valid]

       # True positive detections
    TP = unique(ind_ref).size

    if type(index) is not int:
        # False positive detections
        FP = index.size-unique(ind_ref).size
        # False negative
        FN = index_ref.size-unique(ind_ref).size

    # Detected events identified as True Positive events
    TP_ind = []
    TP_pos = []
    TP_ind = []
    if ind_ref.size>0:
        TP_pos = empty(unique(ind_ref).size)
        TP_ind = empty(unique(ind_ref).size)
        for i, k in enumerate(unique(ind_ref)):
            # ref events indices yielding positive detection
            tmp = where(ind_ref==k)[0]
            # compare values to check who has the strongest response
            max_pos = argmax(value[ref_valid[tmp]])
            pos = ref_valid[tmp[max_pos]]
            TP_pos[i] = pos
            TP_ind[i] = index[pos]

    return TP, FN, FP, TP_pos, TP_ind


def evaluate_detection_old(list_of_detections, true_events, margin = 512):
    """Takes 2 arrays as inputs.
    Returns valid events that have been identified in list_of_detections as present in the ground truth true_events.
    Comparisons are made up to some margin in terms of number of samples around true events.
    Returns a list of Boolean valid_events."""

    nb_detec = size(list_of_detections)

    if nb_detec>0:
        valid_events = [False]*nb_detec
        ind_events = []

        for i, evt in enumerate(list_of_detections):
            valid_events[i] = max(abs(evt-true_events)<margin)
            ind_events.append(int(argmax(abs(evt-true_events)<margin)))
    else:
        valid_events = [False]
        ind_events.append(-1)

    return valid_events, ind_events


def show_detection_bokeh(z, events, Fs=2205, threshold=0.5, lagmin=100):
    """ Detection and interactive graphics
    Detection by thresholding signal z using threshold_detection.py at a variable threshold
    starting from init_threshold. This function is interactive (slider to
    choose the threshold) if interact=True (default)
    Inputs :
        z : the signal to be thresholded
        Fs : sampling freequency, default is Fs = 22050
        init_threshold : the default threshold (should be between 0 and 1)
        lagmin : minimal time lag between 2 successive detection instants (to avoid multiple detections)
    """

    instants = arange(0,z.size/Fs,1/Fs)
    #threshold = init_threshold

    M = max(abs(z))
    #lagmin = round(Fs/20)

    # Detection
    # events = threshold_detection(z_y0, Fe, threshold, lagmin)
    #print(size(events[0]))

    # Graphics
    p = bkfigure(title="Thresholding visualization", plot_height=400, plot_width=600,
           x_range=(instants[0], instants[-1]), y_range=(-M,M), x_axis_label='t [s]');

    sig = p.line(instants, z);
    thresh_line = p.line([0, instants[-1]],[threshold, threshold], color="olive",line_width=2,name="threshline");

    #detec_line = [None]*size(events[0])
    #for i,j in enumerate(events[0]):
    #    detec_line[i]=p.line([instants[j], instants[j]], [0, 1.1*M],color="firebrick",line_width=3);
    xs = []
    ys = []
    #for j in events[0]:
        #xs.append([instants[j], instants[j]])
    if type(events[0]) is not int:
        xs = [[instants[j], instants[j]] for j in events[0]]
        ys =[[0, 1.1*M]]*size(events[0])

        detec_line = p.multi_line(xs,ys,color="firebrick",line_width=2, line_alpha=0.5)
    else:
        detec_line = p.multi_line([],[],color="firebrick",line_width=2, line_alpha=0.5)

    bkshow(p)

    return sig, thresh_line, detec_line, p


def update(threshold=4.):
    thresh_line.data_source.data['y'] = [threshold, threshold]
    detect = threshold_detection(z=z_y0, Fs=Fe, threshold=threshold, lagmin = lagmin)
    list_of_detections = array(detect[0])
    valid_events, ind_events = evaluate_detection(list_of_detections, true_events, margin = lagmin*2)

    ## MULTILINE
    xs = []
    ys = []
    if detect[0]!=-1:   # -1 means NO DETECTION
        for j in detect[0]:
            xs.append([instants[j], instants[j]])
            ys.append([0, 1.1*M])
    # ici utiliser ColumnDataSource plutôt... pour eviter les warnings
    detec_line.data_source.data['xs'], detec_line.data_source.data['ys'] = xs, ys

    # print('xs : ' + str(size(xs)) + ' ys : ' + str(size(ys)))
    print('Nombre de détections : ' + str(len(xs)))
    print('Vrais positifs : ' + str(sum(valid_events)))
    print('Faux positifs : ' + str(size(valid_events)-sum(valid_events)))
    push_notebook()


def update_noisy(threshold=1.):
    thresh_line.data_source.data['y'] = [threshold, threshold]
    detect = threshold_detection(z=z_noise, Fs=Fe, threshold=threshold)
    list_of_detections = array(detect[0])
    valid_events, ind_events = evaluate_detection(list_of_detections, true_events)

    ## MULTILINE
    xs = []
    ys = []
    if detect[0]!=-1:   # -1 means NO DETECTION
        for j in detect[0]:
            xs.append([instants[j], instants[j]])
            ys.append([0, 1.1*M])
    # ici utiliser ColumnDataSource plutôt... pour eviter les warnings
    detec_line.data_source.data['xs'], detec_line.data_source.data['ys'] = xs, ys

    # print('xs : ' + str(size(xs)) + ' ys : ' + str(size(ys)))
    print('Nombre de détections : ' + str(len(xs)))
    print('Vrais positifs : ' + str(sum(valid_events)))
    print('Faux positifs : ' + str(size(valid_events)-sum(valid_events)))
    push_notebook()



"""
def show_detection(z, Fs=22050, threshold = 0.5, interact = True, lagmin = 256):
    Detection + interactive graphics
    Detection by thresholding signal z using threshold_detection.py at a variable threshold
    starting from init_threshold. This function is interactive (slider to
    choose the threshold) if interact=True (default)
    Inputs :
        z : the signal to be thresholded
        Fs : sampling freequency, default is Fs = 22050
        init_threshold : the default threshold (should be between 0 and 1)
        interact : optional interactive display with a slider to explore various thresholds
        lagmin : minimal time lag between 2 successive detection instants (to avoid multiple detections)


    instants = arange(0,z.size/Fs,1/Fs)
    #threshold = init_threshold
    M = max(abs(z))

    def f(threshold):   # for interaction
        # Detection
        events = threshold_detection(z, Fs, threshold, lagmin)

        # Graphics
        fig, ax = plot(instants, z)                # plot the signal to be thresholded
        plot([0, instants[-1]],[threshold, threshold],'r--')  # visualize the threshold
        ax.setp(fontsize=18,xlim=[instants[0], instants[-1]])
        xlabel('t [s]')

        for i in events.size:
            plot([events[i], events[i]], [0, 1.1*M],'m-')

        show()

    interactive_plot = interactive(f, threshold=(0,1.2*M))
    output = interactive_plot.children[-1]
    #output.layout.height = '350px'
    interactive_plot
"""
