from pickle import load
# import tensorflow.compat.v1 as tf  #(it will import tensorflow 1.0 version in your system)
import tensorflow as tf
# from tensorflow.keras.models import Model
from keras.models import load_model
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras import backend as K
#from keras.backend.tensorflow_backend import set_session
from tensorflow.compat.v1.keras.backend import set_session
import numpy as np
import re
import tensorflow
from jiwer import wer

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.compat.v1.Session(config=config)
set_session(sess)
print("use-gpu:", tf.test.gpu_device_name())
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

SEQ_LENGTH = 10
CORRECT_THRESHOLD = 0.001
dump_file_model_left = "./model/model2.h5"
dump_file_model_right = "./model/modelright2.h5"

def text_cleaner(text):
    # lower case text
    newString = text.lower()
    newString = re.sub(r"'s\b","",newString)
    #newString = re.sub("[\.\,!@#$%^&*()[]{}/\|`~-=_+]", "", newString)
    #newString = re.sub(r'\d',"<num>", newString)
    # INTAB = "ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ"
    newString = re.sub("[^a-zA-Zðơ̆ẽĕếĕễê̆¡imĩmĭợơồŏốơ̆ỗô̆ơiơĭổiôĭšĕŠê̆ủŭỦŬũŭŨŬíĭặăãăữư̆ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđàáâãäéêìíïðóôúýăđĕĩĭŏũŭơư̆ạảấầậắằặẹẽếềểễệỉịốổỗộớờỡợụủữỹ']", " ", newString)
    # newString = re.sub("[^a-zA-Zðơ̆ẽĕếĕễê̆imĩmĭợơồŏốơ̆ỗô̆ơiơĭổiôĭšĕŠê̆ủŭỦŬũŭŨŬíĭặăãăữư̆ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđàáâãäéêìíïðóôúýăđĕĩĭŏũŭơư̆ạảấầậắằặẹẽếềểễệỉịốổỗộớờỡợụủữỹ']", " ", newString)
    
    #print('newString:',newString)
    long_words=[]
    # remove short word
    for i in newString.split():
        if len(i)>=2:
            long_words.append(i)
    return (" ".join(long_words)).strip()
def encode_string(mapping, seq_length, in_text):
	# encode the characters as integers
	encoded = [mapping[char] for char in in_text]
	# truncate sequences to a fixed length
	encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
	return encoded

def decode_string(mapping, in_text):
	out_text = ""
	for i in range(len(in_text)):
		for char, index in mapping.items():
			if index == in_text[i]:
				out_text += char
				break
	return out_text


def insert(source_str, insert_str, pos):
    return source_str[:pos]+insert_str+source_str[pos:]

def replace(source_str, insert_str, start_pos):
	source_list = list(source_str)
	if (start_pos > len(source_list)):
		return source_str
	for i in range(len(insert_str)):
		source_list[start_pos + i] = insert_str[i]
	return ''.join(source_list)


# load the model
#model_left = load_model('model_left.h5')

model_left = load_model('model_left.h5')
# model_left = load_model('./model/model2.h5')
# model_right = load_model('./model/modelright2.h5')
model_right = load_model('model_right.h5')
# load the mapping
mapping = load(open('./model/name_data_mapping.pkl', 'rb'))

# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)

print("Number of left layers: %d" % len(model_left.layers))
print("Number of right layers: %d" % len(model_right.layers))

idx_char_mapping = dict([(value, key) for key, value in mapping.items()]) 
SEQ_LENGTH = 10
CORRECT_THRESHOLD = 0.001
print(idx_char_mapping);
print(len(idx_char_mapping))

def next_char_multi_choice(model, mapping, seq_length, seed_text, num=5):
    """
    """
    in_text = text_cleaner(seed_text)
    out_text = in_text[:]
    i = len(in_text)

    out_text_predict_encode = encode_string(mapping , seq_length , out_text)
    proba_list_char = model.predict_on_batch(out_text_predict_encode)

    index_char_list = np.argpartition(proba_list_char, -num)[0,-num:]
    index_char_list = index_char_list[np.argsort(proba_list_char[0][index_char_list])][::-1]

    proba_results = proba_list_char[0, index_char_list]

    char_results = [idx_char_mapping[idx] for idx in index_char_list]

    return char_results, proba_results

#Yôl tơ'nglaih kơpô lơ̆m tơmăn 'nhăt jê̆ 'bă
#Sư̆ pơtho khan nă ma choh jang rim adrêch 'ba la
#'Boi thu 'yŏk điêu tra ra soat ŭnh hnam dơnuh atŭc
# while True:
#     inputText = input('Input : ')
#     output = text_cleaner(inputText)
#     if (inputText == 'Exit'): break
#     char_results, proba_results = next_char_multi_choice(model_left, mapping, SEQ_LENGTH,output, num=5)
#     for idx, char in enumerate(char_results):
#         print(f"output {idx+1}: {output + char} - prob: {proba_results[idx]}")
#     print()
    
####################
#FUNCTION FOR CORRECT MISTAKE oneCHOICE
####################
def correct_one_mistake(model, mapping, input_text):
    in_text = text_cleaner(input_text)
    out_text_predict = in_text[0:5]
    i = 5
    results = ""
    proba_results = 0.0
    prob_list = []
    flag = True
    while True:
        out_text_predict_encode = encode_string(mapping , SEQ_LENGTH, out_text_predict)
        proba_list_char = model.predict_on_batch(out_text_predict_encode)
        predict_x = model.predict(out_text_predict_encode, verbose=0)
        next_char = np.argmax(predict_x, axis=1)
        prob_list.append(proba_list_char[0][mapping[in_text[i]]])
        if((i+1 <= len(in_text)-1) and int(next_char) != mapping[in_text[i]]):
            if(proba_list_char[0][mapping[in_text[i]]] > CORRECT_THRESHOLD):
                # out_text += in_text[i]
                pass
            else:
                proba_correct = proba_list_char[0][mapping[in_text[i]]]
                correct_char = np.argsort(proba_list_char[0])[-1:][::-1]
                proba_results = proba_list_char[0, correct_char]
                results = out_text_predict + decode_string(mapping,correct_char) + in_text[i+1:]
        else:
            # out_text+=in_text[i]
            pass
    
        if(i < len(in_text)-1):
            out_text_predict += in_text[i]
        i = i + 1
        for idx in prob_list:
            if idx < 0.0001:
                flag = False
        else:
            break
    return results, float(proba_results), flag


####################
#FUNCTIONS FOR COMBINE 2 MODEL LEFT AND RIGHT
####################
def correct_mistake(mapping, input_text):
    in_text = text_cleaner(input_text)
    out_text = in_text
    exit_condition = False
    res = []
    i = 1
    while True:
        print('\n')
        print('================== STEP ',i, '=====================')
        in_text = out_text
        result_left, prob_left, flag_left = correct_one_mistake(model_left, mapping, in_text)
        print('Model left: ', result_left, '\t Prob: ',prob_left)
        result_right, prob_right, flag_right = correct_one_mistake(model_right, mapping, in_text[::-1])
        print('Model right: ', result_right[::-1], '\t Prob: ',prob_right)

        if float(prob_left) > float(prob_right):
            out_text = str(result_left)
            print('Choose model left: ',out_text)
            res.append(out_text)
        elif float(prob_left) < float(prob_right):
            out_text = str(result_right[::-1])
            print('Choose model right: ', out_text)
            res.append(out_text)
        
        if (flag_left == True or flag_right == True) or i==20:
            print('End')
            exit_condition = True
        if exit_condition == True:
            break  

        i = i + 1

    if len(res) >=1:
        return str(res[-1])
    else:
        return in_text
#### function look-ahead
def get_p_of_next_char(model, mapping, input_text,lst_lookahead, current_char, next_char, wrong_prob):
    res = {}
    print('Detect : ', input_text , '(',current_char, ')','\t with prob : ',str("{0:.8f}".format(wrong_prob)))
    for e in lst_lookahead:
        out_text = input_text[:] + e
        out_text_predict_encode = encode_string(mapping , SEQ_LENGTH , out_text)
        proba_list_char = model.predict_on_batch(out_text_predict_encode)
        nxt_char = mapping[next_char]
        proba_results = proba_list_char[0, nxt_char]
        res[mapping[e]] = proba_results
    sorted_res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1],reverse=True)}
    char_with_highest_prob = next(iter(sorted_res.items()))
    return char_with_highest_prob
####################
#FUNCTION FOR CORRECT MISTAKE USING LOOKAHEAD
####################
def correct_one_mistake_with_lookahead(model, mapping, input_text):
    in_text = text_cleaner(input_text)
    out_text_predict = in_text[0:5]
    i = 5
    results = ""
    prob_list = []
    flag = True
    while True:
        lst = []
        out_text_predict_encode = encode_string(mapping , SEQ_LENGTH, out_text_predict)
        proba_list_char = model.predict_on_batch(out_text_predict_encode)
        
        prob_list.append(proba_list_char[0][mapping[in_text[i]]])
        predict_x = model.predict(out_text_predict_encode, verbose=0)
        next_char = np.argmax(predict_x, axis=1)
        if((i+1 <= len(in_text)-1) and int(next_char) != mapping[in_text[i]]):
            if(proba_list_char[0][mapping[in_text[i]]] > CORRECT_THRESHOLD):
                # out_text += in_text[i]
                pass
            else:
                wrong_prob = proba_list_char[0][mapping[in_text[i]]]
                index_char_list = np.argsort(proba_list_char[0])[-10:][::-1]
                proba_to_check = proba_list_char[0, index_char_list]
                prob_by_char = dict(zip(index_char_list,proba_to_check))
                correct_char = np.argsort(proba_list_char[0])[-1:][::-1]
                proba_results = proba_list_char[0, correct_char]
                for element in prob_by_char.items():
                # # if element[1] > 0.05:
                # if element[1] > 0.0001:
                    lst.append(element[0])
                lst_lookahead = [in_text[i]]
                lst_lookahead += [idx_char_mapping[e] for e in lst]
                res_lookahead = get_p_of_next_char(model, mapping, out_text_predict,lst_lookahead,in_text[i],in_text[i+1], wrong_prob)

                if res_lookahead[1] <= 0.5:
                    results = out_text_predict + in_text[i:]
                else:
                    results = out_text_predict + str(idx_char_mapping[res_lookahead[0]]) + in_text[i+1:]
        else:
            pass

        if(i < len(in_text)-1):
            out_text_predict += in_text[i]
            i = i + 1
            for idx in prob_list:
                if idx < 0.0001:
                    flag = False
        else:
            break
    return results, str("{0:.8f}".format(wrong_prob)), flag    
# sưa loi 2 mo hinh
#Yôl tơ'nglaih kơpô lơ̆m tơmăn 'nhit jê̆ 'bău
#Sư̆ pơtho khan nă ma choh jang rim adrêch 'ba la
#'Boi thu 'yŏk điêu tra ra soat ŭnh hnam dơnuh atŭc
while True:
    inputText = input('Input: ')
    if (inputText == 'Exit'): break
  
    result = correct_mistake(mapping,inputText)
    print('================== Result =====================')
    print('Input :', inputText)
    print('Output :', result)
    print('\n')

#Yôl tơ'nglaih kơpô lơ̆m tơmăn 'nhit jê̆ 'bău
#Sư̆ pơtho khan nă ma choh jang rim adrêch 'ba la
#'Boi thu 'yŏk điêu tra ra soat ŭnh hnam dơnuh atŭc
# while True:
#     inputText = input('Input : ')
#     output = inputText
#     if (inputText == 'Exit'): break
#     a,b,c = correct_one_mistake_with_lookahead(model_left, mapping,inputText)
#     print(f"output : {a}")
#     print()


#######################################mo hinh look ahead###################

# CHAR_AS_NOTSTRING = '$'
# CHAR_AS_START_SENTENCE = '{'
# CHAR_AS_END_SENTENCE = '}'
# CHAR_AS_COMMA = ','
# CHAR_AS_MINUS = '-'
# CHAR_AS_SPACE = ' '
# CHAR_AS_DOT = '.'
# CHAR_AS_NOTEXISTSTR = '*'

# def text_cleaner(text):
#     # lower case text
# 	newString = text.lower()
# 	newString = re.sub(r"'s\b","",newString)

# 	# remove punctuations
# 	newString = re.sub("[^a-zA-Z{}a-zA-Zðơ̆ẽĕếĕễê̆¡imĩmĭợơồŏốơ̆ỗô̆ơiơĭổiôĭšĕŠê̆ủŭỦŬũŭŨŬíĭặăãăữư̆ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđàáâãäéêìíïðóôúýăđĕĩĭŏũŭơư̆ạảấầậắằặẹẽếềểễệỉịốổỗộớờỡợụủữỹ']", " ", newString)
# 	long_words=[]
# 	# remove short word
# 	for i in newString.split():
# 		if len(i)>=2:
# 			long_words.append(i)
# 	process_str = (" ".join(long_words)).strip()
# 	return process_str

# def encode_string(mapping, seq_length, in_text):
# 	encoded = [mapping[char] for char in in_text]
# 	encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')

# 	return encoded

# def decode_string(mapping, in_text):
# 	out_text = ""
# 	for i in range(len(in_text)):
# 		for char, index in mapping.items():
# 			if index == in_text[i]:
# 				out_text += char
# 				break
# 	return out_text

# def N_max(probs, N): 
# 	listProbs = probs.copy()
# 	final_list = [] 
# 	final_prob = [] 
  
# 	for i in range(0, N):  
# 		max1 = 0
# 		index = 0
# 		for j in range(len(listProbs)):   
# 			if j in range(1,7):
# 				continue 
# 			if listProbs[j] > max1: 
# 				max1 = listProbs[j]
# 				index = j
# 		final_prob.append(max1)                  
# 		listProbs[index] = 0
# 		final_list.append(index) 

# 	return zip(final_list,final_prob)

# def get_probations_next_char(model, mapping, seq_length, current_text):
#     current_text_predict_encode = encode_string(mapping , seq_length , current_text)
#     proba_list_char = model.predict_on_batch(current_text_predict_encode)
#     predict_x = model.predict(current_text_predict_encode, verbose=0)
#     next_char = np.argmax(predict_x, axis=1)
#     return (proba_list_char[0], next_char)



# def get_proba_str(p):
# 	return '0.000000' if p == 0 else "{:.6f}".format(p)

# def string_withspace(s):
# 	return (s + '').replace(' ','_')


# SKIP_FIRST_NCHAR= 3
# DEEP_LOOKAHEAD = 3
# TOPCHECK = 3
# SEQ_LENGTH=20

# def get_top_char_proba(model, mapping, len_seq, text, origin_char):
# 	next_probs = get_probations_next_char(model, mapping, len_seq, text)[0]
# 	top_proba = N_max(next_probs, 10)
# 	array_top = [ (decode_string(mapping,[p[0]]), p[1], p[0]) for p in top_proba]
# 	origin_char_prob = next_probs[mapping[origin_char]] if len(origin_char) > 0 else 0
# 	return (array_top,origin_char_prob)

# def look_ahead_nitem(model, mapping, current_text, list_rest_char, deep, width, current_proba):
#     if deep == 0 or len(list_rest_char) == 0:
#         return []

#     next_origin_text = list_rest_char[0]
#     word_probas, origin_proba = get_top_char_proba(model, mapping, SEQ_LENGTH, current_text, next_origin_text)

#     top_probas = word_probas[:width]

#     char_all = []
#     new_prob = (current_text, next_origin_text, ''.join(list_rest_char[1:]), origin_proba, current_proba, origin_proba * current_proba, deep)
#     # print('new_prob:', new_prob)
#     char_all += [new_prob]

#     # case 1: next_word is redundant, next_2word is ok
#     w_origin_c1 = look_ahead_nitem(model, mapping, current_text, list_rest_char[1:], deep - 1, width, origin_proba)
#     # case 1: next_word is ok, next_2word is ok
#     w_origin_c2 = look_ahead_nitem(model, mapping, current_text + next_origin_text, list_rest_char[1:], deep - 1, width, origin_proba)
#     char_all += w_origin_c1
#     char_all += w_origin_c2
#     for i in range(0,width):
#         # topi-c1: next_word is redundant, replaced by top_probas[i][0]
#         w_topi_c1 = look_ahead_nitem(model, mapping, current_text + top_probas[i][0], list_rest_char[1:], deep - 1, width, top_probas[i][1])
#         # topi-c1: next_word is top_probas[i][0], next_2c is next_char : insert correct to next_char
#         w_topi_c2 = look_ahead_nitem(model, mapping, current_text + top_probas[i][0], list_rest_char, deep - 1, width, origin_proba)

#         char_all += w_topi_c1
#         char_all += w_topi_c2
#     return char_all			

# FINAL='';
# #quận thủ đc thnh ph hồch inh
# while True:
#     inputText = input('Input: ')
#     if (inputText == 'Exit'): break
#     iStep = 0
#     process_text = text_cleaner(inputText)
#     while True:
#         iStep += 1
#         print("---------------------------------------------")
#         print('STEP:' + str(iStep) + ':' + process_text)
#         before_correct = process_text + ''
#         for i in range(SKIP_FIRST_NCHAR, len(process_text) - 1):
#             current_text = process_text[:i]
#             print(f">> {current_text}({process_text[i]})")
#             probas = get_top_char_proba(model_left, mapping, SEQ_LENGTH, current_text, process_text[i])
#             if probas[1] > CORRECT_THRESHOLD:
#                 # print(f"<< '{process_text[i]}' is good enough ({get_proba_str(probas[1])})")
#                 continue
#             else:
#                 # # ONLY REPLACE A CHARACTER
#                 # print(f"<< '{process_text[i]}' is not good ({get_proba_str(probas[1])} < {CORRECT_THRESHOLD})")
#                 # print(f"<< Best Match: '{probas[0][0]}' {get_proba_str(probas[0][1])}")
#                 # process_text = current_text + probas[0][0] + process_text[i+1:]
#                 # USING LOOK AHEAD
#                 rest_text = [ x for x in process_text[i:] ]
#                 all_cases = look_ahead_nitem(model_left, mapping, current_text, rest_text , DEEP_LOOKAHEAD, TOPCHECK, 1)
#                 print(all_cases)
#                 dict_result = {}
#                 # [ print(x) for x in all_cases]
#                 for x in all_cases:
#                   key = f"{x[0]}{x[1]}{x[2]}"
#                   if key in dict_result and dict_result[key][0] > x[5]:
#                     continue
#                   dict_result[key] = (x[5], x[6])

#                 all_dict_cases = [ (x, dict_result[x]) for x in dict_result]
#                 all_dict_cases.sort(key=lambda tup: tup[1][0], reverse=True)
#                 [ print(f"<< [top look ahead] {x[0]} ({get_proba_str(x[1][0])})") for x in all_dict_cases[0:5]]

#                 process_text = all_dict_cases[0][0]
#                 break
#         if before_correct == process_text:
#             # nothing to correct
#             break
#     #yôl adrĭng 'bă 'băn nb x tơman
#     #huyên vxnh thaavh
#     #huyêên vxnh than
#     #correct


#     print("=============================================")
#     print('[INPUT] ' + inputText)
#     print("---------------------------------------------")
#     print('[PREDICT LOOKAHEAD] ' + process_text)
#     output = process_text
#     if (inputText == 'Exit'): break
    