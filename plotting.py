
import matplotlib.pyplot as plt
import numpy as np
import cv2

images = []
name = 'knn'

for i in range(5):
  image_path = 'figures/k_fold/'+ name +'_conf_graph_'+str(i+1)+'.png'
  img = cv2.imread(image_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  images.append(img)

img = cv2.imread('figures/k_fold/'+ name +'_overlapped_matrix_graph.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
images.append(img)

plt.rcParams["figure.figsize"] = (18,10)
f, axarr = plt.subplots(2,3)
axarr[0,0].imshow(images[0])
axarr[0,0].set_title('Fold 1')
axarr[0,1].imshow(images[1])
axarr[0,1].set_title('Fold 2')
axarr[0,2].imshow(images[2])
axarr[0,2].set_title('Fold 3')
axarr[1,0].imshow(images[2])
axarr[1,0].set_title('Fold 4')
axarr[1,1].imshow(images[3])
axarr[1,1].set_title('Fold 5')
axarr[1,2].imshow(images[4])
axarr[1,2].set_title('Overlapped Matrix')


images = []
name = 'knn'

for i in range(5):
  image_path = 'figures/k_fold/roc_curve_'+ name +'_fold_'+ str(i+1) +'.png'
  img = cv2.imread(image_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  images.append(img)


plt.figure(figsize=(30,30)) 

for i in range(3):
    plt.subplot(5,5,i+1)    
    plt.title('Fold ' + str(i+1))
    plt.imshow(images[i])

plt.show()
plt.figure(figsize=(30,30)) 
for i in range(3,5):
    plt.subplot(5,5,i+1)   
    plt.title('Fold ' + str(i+1))
    plt.imshow(images[i])

plt.show()




images = []
name = 'mycnn'

for i in range(5):
  image_path = 'figures/k_fold/'+ name +'_acc_loss_graph_fold_'+ str(i+1) +'.png'
  img = cv2.imread(image_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  images.append(img)


plt.figure(figsize=(60,40)) 

for i in range(2):
    plt.subplot(5,5,i+1)    
    plt.title('Fold ' + str(i+1))
    plt.imshow(images[i])

plt.show()
plt.figure(figsize=(60,40)) 
for i in range(2,4):
    plt.subplot(5,5,i+1)   
    plt.title('Fold ' + str(i+1))
    plt.imshow(images[i])

plt.show()

plt.figure(figsize=(60,40)) 
for i in range(4, 5):
    plt.subplot(5,5,i+1)   
    plt.title('Fold ' + str(i+1))
    plt.imshow(images[i])

plt.show()