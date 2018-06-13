import numpy as np
import os
import math
import time
# Opening the files

time_begin= time.time()
os.chdir( "C:/Users/Mergen/Desktop/CS440/MP3 part1 CS440/" )

train_labels= open( "traininglabels", "r" )
train_images= open( "trainingimages", "r" )

train_image_lines =train_images.readlines()
train_label_lines = train_labels.readlines()
number_of_train = len( train_label_lines )
k=5

# Digit reader
def Read_digit(position, type):
    if( type=="training"):
        image_lines= train_image_lines
        label_lines= train_label_lines
    else:
        image_lines=test_image_lines
        label_lines=test_label_lines

    Digit_matrix=[]
    for i in range( 28*(position), 28*(position+1), 1):
        line_array = list( image_lines[i] )

        #convert list(string) to list(digit)
        Digit_array=[]
        for s in range(0, len(line_array)-1 ):
            if( line_array[s]==" "):
                Digit_array.append(0)
            else:
                Digit_array.append(1)
        # add row to the digit_matrix
        Digit_matrix.append(Digit_array)

    label = int( label_lines[ position] )

    return Digit_matrix, label


# ---------------------- Training ---------------------------------------------------
# Initialize variable
probability_f_c=np.zeros( (10,28,28) )    # (class, row, col)
actual_digit_freq = np.zeros( 10)

for i in range( number_of_train ):
    Digit_matrix, label= Read_digit(i, "training") 
    Digit_array = np.array( Digit_matrix)
    probability_f_c[label] +=Digit_array
    actual_digit_freq[label] +=1

for i in range(10):
    probability_f_c[i] = ( probability_f_c[i] + k) / ( actual_digit_freq[i] + 2*k)

probability_class= actual_digit_freq / number_of_train

# ------------------ Testing --------------------------------------------------------
train_labels.close()
train_images.close()
test_labels= open( "testlabels", "r" )
test_images= open( "testimages", "r" )

test_image_lines =test_images.readlines()
test_label_lines = test_labels.readlines()
number_of_test = len( test_label_lines )

def decode_image( pos):
    Digit_matrix, true_label = Read_digit(pos, "test")
    probability_class_F= np.zeros(10)

    for i in range(10):
        probability = math.log10( probability_class[i] )
        for x in range(28):
            for y in range(28):
                if( Digit_matrix[y][x]==1):
                    probability += math.log10( probability_f_c[i][y][x] )
                else:
                    probability += math.log10( 1-probability_f_c[i][y][x] )
        probability_class_F[i] = probability
    calculated_label = np.argmax( probability_class_F) 

    return true_label, calculated_label

# print(decode_image(1) )
# print( decode_image(1))
# print( decode_image(2))
# print( decode_image(3))
# print( decode_image(4))

true_freq=np.zeros(10)
calculated_freq=np.zeros(10)
confusion_matrix=np.zeros( (10,10) )
true_matrix=np.zeros( (10,10) )

for i in range( number_of_test):
    true_label, calculated_label = decode_image(i)
    true_freq[true_label] +=1
    confusion_matrix[true_label, calculated_label]+=1

for r in range(10):
    confusion_matrix[r]= 100*confusion_matrix[r]/true_freq[r]

correct_classification_rate= [ confusion_matrix[i][i] for i in range(10)]

time_end = time.time()
runtime = time_end - time_begin
test_labels.close()
test_images.close()

print("Laplacian smoothing, k= ", k)
print( """-----------Correct classification rate--------------""")
print( correct_classification_rate)

print( """----------- Confusion Matrix--------------""")
print(confusion_matrix)

print( "Runtime: " , runtime, " seconds")


#------------------   Finding maximumal Odd-pairs----------------------------
for i in range(10):
    confusion_matrix[i][i]=-9999
odd_pairs=[]
for i in range(4):
    index = np.argmax( confusion_matrix)
    maxc = index % 10
    maxr = int( index/10 )
    odd_pairs.append( [maxr,maxc])
    confusion_matrix[maxr][maxc]= -9999

print( "Max confusion:" , odd_pairs)

Odd_ratio_matrices = np.zeros( (4,28,28) )
for i in range(4):
    classA= odd_pairs[i][0]
    classB= odd_pairs[i][1]
    Odd_ratio_matrices[i] = np.log10( probability_f_c[classA] / probability_f_c[classB] )

### ---------------- Generating log likelihood graphs ---------------------------
import matplotlib.pyplot as plt

x = np.linspace(1, 28, 28) 
y = np.linspace(1, 28, 28)
X,Y = np.meshgrid(x,y)
Z_0 = Odd_ratio_matrices[0]
Z_1 = Odd_ratio_matrices[1]
Z_2 = Odd_ratio_matrices[2]
Z_3 = Odd_ratio_matrices[3]

fig = plt.figure(figsize = (12,2.5) , dpi=300)
fig.subplots_adjust(wspace=0.3)

# ----------------------------------- Plotting odd-pair 1
numberA = odd_pairs[0][0]
numberB = odd_pairs[0][1]
title1 = 'Similarity of classes ' + str( numberA ) +  "and " + str( numberB )
plt.subplot(1,4,1)
plt.pcolormesh(X, Y, Z_0, cmap=plt.cm.get_cmap('seismic'))
plt.colorbar()
plt.axis([1,28, 1,28])
plt.title(title1 )
plt.gca().invert_yaxis()
plt.show()


# Blues
title1a = 'Probability of class ' + str( numberA )
plt.subplot(1,4,2)
plt.pcolormesh(X, Y, probability_f_c[ numberA ], cmap=plt.cm.get_cmap('seismic'))
plt.colorbar()
plt.axis([1,28, 1,28])
plt.title(title1a )
plt.gca().invert_yaxis()
plt.show()

# Blues
title1b = "Probability of class "+ str( numberB )
plt.subplot(1,4,3)
plt.pcolormesh(X, Y, probability_f_c[numberB] , cmap=plt.cm.get_cmap('seismic'))
plt.colorbar()
plt.axis([1,28, 1,28])
plt.title(title1b )
plt.gca().invert_yaxis()
plt.show()

# ----------------------------------- Plotting odd-pair 2

numberA = odd_pairs[1][0]
numberB = odd_pairs[1][1]
title1 = 'Similarity of classes ' + str( numberA ) +  "and " + str( numberB )
plt.subplot(2,4,1)
plt.pcolormesh(X, Y, Z_1, cmap=plt.cm.get_cmap('seismic'))
plt.colorbar()
plt.axis([1,28, 1,28])
plt.title(title1 )
plt.gca().invert_yaxis()
plt.show()


# Blues
title1a = 'Probability of class' + str( numberA )
plt.subplot(2,4,2)
plt.pcolormesh(X, Y, probability_f_c[ numberA ], cmap=plt.cm.get_cmap('seismic'))
plt.colorbar()
plt.axis([1,28, 1,28])
plt.title(title1a )
plt.gca().invert_yaxis()
plt.show()

# Blues
title1b = "Probability of class "+ str( numberB )
plt.subplot(2,4,3)
plt.pcolormesh(X, Y, probability_f_c[numberB] , cmap=plt.cm.get_cmap('seismic'))
plt.colorbar()
plt.axis([1,28, 1,28])
plt.title(title1b )
plt.gca().invert_yaxis()
plt.show()


# ----------------------------------- Plotting odd-pair 3

numberA = odd_pairs[2][0]
numberB = odd_pairs[2][1]
title1 = 'Similarity of classes ' + str( numberA ) +  "and " + str( numberB )
plt.subplot(3,4,1)
plt.pcolormesh(X, Y, Z_2, cmap=plt.cm.get_cmap('seismic'))
plt.colorbar()
plt.axis([1,28, 1,28])
plt.title(title1 )
plt.gca().invert_yaxis()
plt.show()


# Blues
title1a = 'Probability of class ' + str( numberA )
plt.subplot(3,4,2)
plt.pcolormesh(X, Y, probability_f_c[ numberA ], cmap=plt.cm.get_cmap('seismic'))
plt.colorbar()
plt.axis([1,28, 1,28])
plt.title(title1a )
plt.gca().invert_yaxis()
plt.show()

# Blues
title1b = "Probability of class "+ str( numberB )
plt.subplot(3,4,3)
plt.pcolormesh(X, Y, probability_f_c[numberB] , cmap=plt.cm.get_cmap('seismic'))
plt.colorbar()
plt.axis([1,28, 1,28])
plt.title(title1b )
plt.gca().invert_yaxis()
plt.show()


# ----------------------------------- Plotting odd-pair 4

numberA = odd_pairs[3][0]
numberB = odd_pairs[3][1]
title1 = 'Similarity of classes ' + str( numberA ) +  "and " + str( numberB )
plt.subplot(4,4,1)
plt.pcolormesh(X, Y, Z_3, cmap=plt.cm.get_cmap('seismic'))
plt.colorbar()
plt.axis([1,28, 1,28])
plt.title(title1 )
plt.gca().invert_yaxis()
plt.show()


# Blues
title1a = 'Probability of class ' + str( numberA )
plt.subplot(4,4,2)
plt.pcolormesh(X, Y, probability_f_c[ numberA ], cmap=plt.cm.get_cmap('seismic'))
plt.colorbar()
plt.axis([1,28, 1,28])
plt.title(title1a )
plt.gca().invert_yaxis()
plt.show()

# Blues
title1b = "Probability of class "+ str( numberB )
plt.subplot(4,4,3)
plt.pcolormesh(X, Y, probability_f_c[numberB] , cmap=plt.cm.get_cmap('seismic'))
plt.colorbar()
plt.axis([1,28, 1,28])
plt.title(title1b )
plt.gca().invert_yaxis()
plt.show()















