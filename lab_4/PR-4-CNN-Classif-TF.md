Практикум 4\. Побудова моделi згорткових нейронних мереж для класифiкацiї зображень в TensorFlow

Мета роботи:

• Навчитися створювати моделi згорткових нейронних мереж за допомогою бi блiотеки TensorFlow та API Keras.

• Використати згорткову нейронну мережу для класифiкацiї чорно-бiлих та ко льорових зображень.

• Порiвняти результати з результатами на основi моделi MLP.

• Для представлення результатiв використати TensorBoard.

1 Хiд виконання роботи:

1\. Завантажити два набори зображень згiдно з варiантом: чорно-бiлi (дивiться по переднiй практикум) та кольоровi. Якщо набiр великого розмiру, можна обрати частину.

2\. Пiдготувати данi для навчання \- за необхiдностi.

3\. Роздiлити данi на навчальну, перевiрочну i тестову пiдмножини.

4\. Побудувати i навчити базову модель з одним згортковим шаром. Оцiнити пра вильнiсть (accuracy) та точнiсть моделi на тренувальнiй i перевiрочнiй множи нах.

5\. Дослiдити рiзнi значення параметрiв padding i strides згорткового шару ба зової моделi, а також розмiрнiсть ядра (kernel) згортки та їх вплив на точнiсть моделi. На перевiрочнiй множинi обрати значення вказаних параметрiв.

1  
6\. Дослiдити кiлька альтернативних архiтектур згорткових моделей, якi включа ють:

• кiлька згорткових шарiв,

• шар/ шари нормалiзацiї за мiнi-батчами,

• шар/ шари дропауту.

На перевiрочнiй множинi оцiнити якiсть побудованих моделей i обрати най кращу архiтектуру. Використати показники якостi: accuracy, precision, recall, f1-score, AUC.

7\. Чи впливає на правильнiсть (accuracy) моделi додавання регуляризацiї: дропа ут, рання зупинка навчання, та використання рiзних методiв iнiцiалiзацiї ваг?

8\. Вiдобразити у TensorBoard графiки, якi iлюструють оцiнки якостi навчання мереж на навчальнiй та перевiрочнiй множинах:

\- графiки змiни функцiї втрат на тренувальнiй i перевiрочнiй множинах по мiрi навчання моделей,

\- графiки змiни правильностi моделi на тренувальнiй i перевiрочнiй множинах по мiрi навчання моделей.

9\. Розрахувати на тестовiй множинi оцiнки якостi обраної найкращої моделi.

10\. Завантажити зображення тестової множини i розпiзнати його навченими моде лями.

11\. Порiвняти побудованi згортковi моделi та багатошаровий персептрон в задачi класифiкацiї чорно-бiлих та кольорових зображень.

12\. Зробити висновки щодо якостi класифiкацiї на основi побудованих моделей.

2 Навчальнi данi обирати згiдно з варiантом 3 Теоретичнi вiдомостi

3.1 Побудова та навчання згорткової мережi для класифiкацiї зображень засобами Keras

Розглянемо побудову та навчання згорткової мережi засобами Keras на прикладi класифiкацiї зображень MNIST та CIFAR10, CIFAR100.

12 Shoe vs Sandal vs Boot Image Dataset, kaggle.com

from k e ra s . d a t a s e t s import mnist

from k e ra s . d a t a s e t s import c i f a r 1 0

from k e ra s . d a t a s e t s import c i f a r 1 0 0

from t e n s o rfl o w . k e ra s . models import S e q u e n ti al

3  
from t e n s o rfl o w . k e ra s . l a y e r s import Dense , Conv2D , MaxPooling2D , Fla t ten , Dropout , Ba tchNo rmaliza tion , GlobalAveragePooling2D from t e n s o rfl o w . k e ra s import u t i l s

from t e n s o rfl o w . k e ra s . p r e p r o c e s s i n g import image

from g oo gl e . cola b import f i l e s

import numpy a s np

import m a t pl o tli b . p y plo t a s p l t

from PIL import Image

Завантаження наборiв данних MNIST та CIFAR:

( x\_train , y\_ t rain ) , ( x\_test , y\_ tes t ) \= mnist . load\_data ( ) ( x\_train10 , y\_ t rain10 ) , ( x\_test10 , y\_ tes t10 ) \= c i f a r 1 0 . load\_data ( ) ( x\_train100 , y\_ t rain100 ) , ( x\_test100 , y\_ tes t100 ) \= c i f a r 1 0 0 . load\_data ( )

Переведення еталонних значень (мiток зображень) в формат one-hot:

y\_ t rain \= u t i l s . t o \_ c a t e g o ri c al ( y\_train , 10 )

y\_ t rain10 \= u t i l s . t o \_ c a t e g o ri c al ( y\_train10 , 10 )

y\_ t rain100 \= u t i l s . t o \_ c a t e g o ri c al ( y\_train100 , 100 ) Виведення одного iз зображень навчальної вибiрки CIFAR-10на екран:

n \= 20

p l t . imshow ( Image . f roma r ra y ( x\_ t rain10 \[ n \] ) . co n v e r t ( ’RGBA’ ) ) p l t . show ( )

pr int ( x\_ t rain10 \[ n \] )

Перетворення кожного вхiдного зображення у двовимiрний масив:

x\_ t rain \= x\_ t rain . r e s ha p e ( x\_ t rain . shape \[ 0 \] , 28 , 28 , 1 ) x\_tes t \= x\_ tes t . r e s ha p e ( x\_ tes t . shape \[ 0 \] , 28 , 28 , 1 )

Створення базової згорткової моделi з одним згортковим шаром i шаром макс пулiнгу:

model \= S e q u e n ti al ( )

model . add (Conv2D (32 , ( 3 , 3 ) , padding=’ same ’ , a c t i v a t i o n=’ r e l u ’ , input\_shape \=(28 , 28 , 1 ) ) )

model . add ( MaxPooling2D ( pool\_ si z e \=(2 , 2 ) ) )

4  
model . add ( Fla t t e n ( ) )

model . add ( Dense (10 , a c t i v a t i o n=’ softmax ’ ) )

Повнозв’язнi шари створюються за допомогою шару Flatten(), який виконує переформатування тензора, аналогiчно до функцiї tf.reshape.

Компiляцiя моделi i виведення результатiв компiляцiї:

model . compile ( l o s s=" c a t e g o r i c a l \_ c r o s s e n t r o p y " , o p timi z e r="adam" , m e t ri c s \=\[" accu rac y " \] )

pr int ( model . summary ( ) )

Модель буде навчати 320 параметрiв на згортковому шарi i 14 *×* 14 *×* 32 \= 6272 параметри на вихiдному шарi softmax:

Model : " s e q u e n t i a l "

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ Layer ( type ) Output Shape Param *\#* \================================================================= conv2d (Conv2D ) ( None , 28 , 28 , 32 ) 320 \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ max\_pooling2d ( MaxPooling2D ) ( None , 14 , 14 , 32 ) 0 \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ f l a t t e n ( Fla t t e n ) ( None , 6272 ) 0 \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ den se ( Dense ) ( None , 10 ) 62730 \================================================================= To tal params : 63 ,050

T rai na bl e params : 63 ,050

Non−t r a i n a b l e params : 0

Запуск моделi на навчання:

h i s t o r y \= model . f i t ( x\_train ,

y\_train ,

ba tch\_ siz e \=200,

epoch s \=20,

v a l i d a t i o n \_ s p l i t \=0.2 ,

v e r bo s e \=1)

В данному прикладi ралiзовано ранню зупинку навчання, використовуючи фун кцiю ModelCheckpoint з модуля keras.callbacks: на кожнiй епосi перевiрялася ме трика якостi \- точнiсть на перевiрочнiй множинi (val\_acc). Пiсля сьомої епохи то чнiсть val\_acc=0.9777 була найбiльша, i використано ваги пiсля цiєї епохи:

5  
Train on 48000 samples , v a l i d a t e on 12000 sample s Epoch 1/20

−l o s s : 1.2027 − acc : 0.9124 − v al\_l o s s : 0.2113 − val\_acc : 0.9650 Epoch 2/20

−l o s s : 0.1534 − acc : 0.9695 − v al\_l o s s : 0.1615 − val\_acc : 0.9693 . . .

Epoch 7/20

−l o s s : 0.0214 − acc : 0.9928 − v al\_l o s s : 0.1270 − val\_acc : 0.9777

Графiки змiни точностi на тренувальнiй i перевiрочнiй множинах по мiрi навчання моделi для всiх двадцяти епох наведено на рис. 1\. З графiку видно, що пiсля сьомої епохи у моделi спостерiгається невелике перенавчання.

![][image1]  
Рис. 1: Графiки змiни точностi на тренувальнiй i перевiрочнiй множинах для базової моделi

Наступна модель мiстить два згорткових шари, доповненi нормалiзацiєю за мiнi батчами i дропаутом. Пiсля згорткових слiдують повнозв’язнi шари: один з актива цiєю ReLU, другий вихiдний з активацiєю SoftMax:

model \= S e q u e n ti al ( )

model . add ( Ba tchNo rmaliza tion ( input\_shape \=(28 , 28 , 1 ) ) ) model . add (Conv2D (32 , ( 3 , 3 ) , padding=’ same ’ , a c t i v a t i o n=’ r e l u ’ ) ) model . add (Conv2D (32 , ( 3 , 3 ) , padding=’ same ’ , a c t i v a t i o n=’ r e l u ’ ) ) model . add ( MaxPooling2D ( pool\_ si z e \=(2 , 2 ) ) )

model . add ( Dropout ( 0 . 2 5 ) )

6  
model . add ( Ba tchNo rmaliza tion ( ) )

model . add (Conv2D (64 , ( 3 , 3 ) , padding=’ same ’ , a c t i v a t i o n=’ r e l u ’ ) ) model . add (Conv2D (64 , ( 3 , 3 ) , padding=’ same ’ , a c t i v a t i o n=’ r e l u ’ ) ) model . add ( MaxPooling2D ( pool\_ si z e \=(2 , 2 ) ) )

model . add ( Dropout ( 0 . 2 5 ) )

model . add ( Fla t t e n ( ) )

model . add ( Dense (512 , a c t i v a t i o n=’ r e l u ’ ) )

model . add ( Dropout ( 0 . 5 ) )

model . add ( Dense (10 , a c t i v a t i o n=’ softmax ’ ) )

Компiляцiю моделi проведено з тими ж параметрами, що i у першiй моделi. На вчання моделi:

h i s t o r y \= model . f i t ( x\_train ,

y\_train ,

ba tch\_ siz e \=200,

epoch s \=20,

v a l i d a t i o n \_ s p l i t \=0.2 ,

v e r bo s e \=1)

показало наступнi результати на навчальнiй i перевiрочнiй множинах:

Train on 48000 samples , v a l i d a t e on 12000 sample s Epoch 1/20

−l o s s : 0.2930 − acc : 0.9111 − v al\_l o s s : 0.1387 − val\_acc : 0.9700 Epoch 2/20

−l o s s : 0.0689 − acc : 0.9789 − v al\_l o s s : 0.0369 − val\_acc : 0.9894 . . .

Epoch 19/20

−l o s s : 0.0167 − acc : 0.9947 − v al\_l o s s : 0.0294 − val\_acc : 0.9940

Графiки змiни точностi на тренувальнiй i перевiрочнiй множинах по мiрi навчання моделi для всiх епох наведено на рис. 2\.

Друга модель показала вищу точнiсть на перевiрочнiй множинi порiвняно з пер шою моделлю.

Використаємо цю другу модель до данних CIFAR10. Оскiльки кожне зображен ня має розмiр 32 *×* 32, треба змiнити розмiрнiсть входу в першому шарi мережi на input\_shape=(32, 32, 3):

model \= S e q u e n ti al ( )

Рис. 2: Графiки змiни точностi на тренувальнiй i перевiрочнiй множинах для розши реної моделi

model . add ( Ba tchNo rmaliza tion ( input\_shape \=(32 , 32 , 3 ) ) ) model . add (Conv2D (32 , ( 3 , 3 ) , padding=’ same ’ , a c t i v a t i o n=’ r e l u ’ ) ) model . add (Conv2D (32 , ( 3 , 3 ) , padding=’ same ’ , a c t i v a t i o n=’ r e l u ’ ) ) model . add ( MaxPooling2D ( pool\_ si z e \=(2 , 2 ) ) )

model . add ( Dropout ( 0 . 2 5 ) )

model . add ( Ba tchNo rmaliza tion ( ) )

model . add (Conv2D (64 , ( 3 , 3 ) , padding=’ same ’ , a c t i v a t i o n=’ r e l u ’ ) ) model . add (Conv2D (64 , ( 3 , 3 ) , padding=’ same ’ , a c t i v a t i o n=’ r e l u ’ ) ) model . add ( MaxPooling2D ( pool\_ si z e \=(2 , 2 ) ) )

model . add ( Dropout ( 0 . 2 5 ) )

model . add ( Fla t t e n ( ) )

model . add ( Dense (512 , a c t i v a t i o n=’ r e l u ’ ) )

model . add ( Dropout ( 0 . 5 ) )

model . add ( Dense (10 , a c t i v a t i o n=’ softmax ’ ) )
