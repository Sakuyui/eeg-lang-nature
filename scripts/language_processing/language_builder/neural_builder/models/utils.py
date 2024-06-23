import numpy as np
def rebuild_T_from_head_left_right(head, left, right, NT, T):
    r_dim = head.shape[1]
    sum_NT_T = NT + T
    T = np.zeros((NT * sum_NT_T * sum_NT_T))
    for r in range(r_dim):
        T += np.kron(np.kron(head[:, r].detach().numpy(), left[:, r].detach().numpy()), right[:, r].detach().numpy())
    return T.reshape((NT, sum_NT_T, sum_NT_T))
    
def out_pretermination(inference_result, cnt_words, word_sequence):
    unary = inference_result['unary']
    T = inference_result['unary'].shape[-1]
    sequence_length = inference_result['unary'].shape[-2]
    print("Preterminate count = %d, Dictionary size = %d, sequence length = %d" % (T, cnt_words, sequence_length))
    matrix = np.zeros((cnt_words, T))
    for i in range(sequence_length):
        matrix[word_sequence[i], :] = unary[0, i, :].detach().numpy()
    s = ""
    for i in range(cnt_words):
        for j in range(T):
            s += ("%s -> %s [%lf]" % \
                  ('P' + str(j), 'w' + str(i), matrix[i, j])) + "\n"
    return s

def out_start(inference_result):
    root = inference_result['root']
    size_Nt = inference_result['root'].shape[-1]
    print("Nt = %d" % (size_Nt))
    s = ""
    for i in range(size_Nt):
        s += ("%s -> %s [%lf]" % \
              ('S', 'S' + str(i), root[0, i])) + "\n"
        
        
def out_double_non_terminates(inference_result):
    head = inference_result['head']
    left = inference_result['left']
    right = inference_result['right']
    Nt = head.shape[-2]
    sum_Nt_t = left.shape[-2]
    s = ""
    n_r = head.shape[-1]
    T = np.zeros((Nt * sum_Nt_t * sum_Nt_t))
    for l in range(n_r):
        T += np.kron(np.kron(head[0, :, l], left[0,:,l]), right[0,:,l])
    for i in range(Nt):
        for j in range(sum_Nt_t):
            for z in range(sum_Nt_t):
                s += ('%s -> %s %s [%lf]' %\
                      ('S' + str(i),
                       'S' + str(j) if j < Nt else 'P' + str(j - Nt),
                       'S' + str(z) if z < Nt else 'P' + str(z - Nt),
                        T[i * (sum_Nt_t * sum_Nt_t) + j * sum_Nt_t + z]
                                   )) + "\n"
    return s