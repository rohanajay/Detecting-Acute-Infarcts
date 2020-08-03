PES UNIVERSITY
CIE Deep Learning Course

FINAL PROJECT : IDENTIFICATION ON ACUTE INFARCT USING CNN MODEL

TEAM NUMBER : 9

TEAM MEMBERS:
	1. Name : NAGAVISHNU B K   SRN : PES1201700421
	2. Name : ROHAN AJAY       SRN : PES1201700454
	3. Name : SANJANA S RAO    SRN : PES1201701514
	4. Name : ASHWIN UMADI     SRN : PES1201701517
        5. Name : KUSHAL N         SRN : PES1201701567

CONTENTS IN ZIP FOLDER:
	1. FILTERED_DATA folder :
		i.) This folder is further divided into a train and test folder.
		|
		|___ a.) train: This folder consists of 38 unique classes(Folders) obtained after dataset
		|		optimisation. It in total contains 45 images that are used for 
		|		training the model.
		|
		|___ b.) test : This folder consists of 5 classes (Folders) with 1 image in each folder.
				Thus in total there are 5 images.
	2. INFERENCE_DATA folder :
		i.) This folder has a total of 23 unique classes (Folders) with 36 images in total.
		    Few classes in this folder are not trained to the model, and there were few classes
		    with slightly modified names even though the model was trained for those classes.
		    So we changed the names of few classes so that it matches to the given when the model 
		    was trained. For example, we renamed
		    (Right fronto-parietal lobes, right caudate nucleus) as (Right fronto-parietal lobes)
	3. Acute_infarct.txt file:
		i.) This is file that gets generated once the train.py (CNN training model) is executed. 
		    It contains the labels of the classes in the order the model is trained so that it can be
		    used during prediction and for the final python application code as well.
	4. train.py file:
		i.) This is the python code file which implements the CNN model using keras on tensorflow background.
		    Once the model is trained, a 'mri_predict.h5' file is generated which is stored in the same location 
		    as that of the source code.
	5. mri_predict.h5 file :
		i.) This is the file that gets generated once the train.py is run. The .h5 file stores the weights of the
		    trained model.
	6. prediction.py file:
		i.) This is the python file to be run once the .h5 file is predicted for obtaining the training, testing and
		    inference image accuracy.
	7. get_IR_from_h5.py file:
		i.) This is the python file that has to be run to convert the 'mri_predict.h5' file to 'mri_predict.pb' (frozen model)
			and then further to a IR representation model. i.e the .xml and .bin file.
		    The script to run this code is explained in below sections.
	8. mri_predict.xml, mri_predict.bin and mri_predict.mapping files:
		i.) These are the IR representation model given as an input to the final python app code.
	9. mri_predict.py file:
		i.) This is the final python application code used to preform the prediction of a given inference image. 
		    The script to run this file is mentioned below and this can to used to make the model run on the NCS-2 toolkit.

PREREQUISITES:
	1.) Python version 3.7.x
	2.) Tensorflow version 1.14/ tensorflow-gpu version 2.0.0 and keras installed.
	3.) opencv, numpy modules installed.
	4.) Intel model optimizer i.e mo.py file being present provided by openvino.

EXECUTION OF THE MODEL:
	1. First direct your command prompt to the current downloaded folder. Opening your command prompt with administrative access is required.
		Type the following command to get it done.
		>>cd <your extracted zipped folder location>

	2. Once done run the following command to run the train.py model
		>>python train.py

	3. After training, to run the prediction model, execute the following code.
		>>python prediction.py
		Your command prompt will now show the accuracy on the training image, testing image and on the inference image.

	4. Now to convert the '.h5' file to a frozen model use the keras_to_tensorflow.py code. Execute the following command
		(Administrative access mandatory)
		>>python get_IR_from_h5.py --input_model "<your extracted zip folder location>\mri_predict.h5" 
			--output_model "<your extraxted zip folder location>\mri_predict.pb".
		Here you can give the output directory wherever you wish by entering that location after '--output_model'.
		for example:
		>>python get_IR_from_h5.py --input_model "C:\Users\ashwin\Documents\CIE_Deep_learning\Final_project\mri_predict.h5" 
				--output_model "C:\Users\ashwin\Documents\CIE_Deep_learning\Final_project\mri_predict.pb"

		This particular code consists of 2 parts,
		a.) Generation of a .pb file (frozen model), generation of an IR model i.e, the .xml and .bin and .mappings file

		Half way after the execution of the get_IR_from_h5.py code, the code asks for the location of the model_optimizer provided by openvino 
		software. There you will have to enter the path to the model optimizer folder.

		The command prompt output would be like as shown below,

		INFO
		Enter the path to your model optimizer file mo_tf.py (Please do not include the file name)
		Please make sure there are no spaces gives at the start
		Please do not enter the path in quotation marks/ inverted commas

		Enter here :

		You will have to type the path infront of the 'Enter here :' message, 
			in our case the path is, "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\model_optimizer"

		Once this is run, you will get a success message saying the .xml and .bin files have been generated in the directory where the frozen model 
		is present.
		

	6. Now to run the python app code, we pass the .xml file, the labels file and the inference image as input and obtain the predicted class as output
		Before running this code you have to initialize the Openvino environment variables. This can be done using the following command
		>>cd '<INSTALLATION DIRECTORY OF OPEN VINO>/bin'
		once the directory is set run the command
		>>setupvars.bat
 
		Now,
		The command prompt should direct to your zipped folder location( location where the python app code is present).
		>>cd "location where the python app code is present"
		  The command script goes as follows
		>>python mri_predict.py --model "<location of the directory where the .xml file is present>\mri_predict.xml" 
				--input "<location of the dicrectory where the inference image is present>\<Image name>" 	
				--labels "<location of the directory where you app code is present (your extracted zipped folder location)>\Acute_infarct.txt"
		for example:
		>>python mri_predict.py --model "C:\Users\ashwin\Documents\CIE_Deep_learning\Final_project\mri_predict.xml"
			 --input "C:\Users\ashwin\Documents\CIE_Deep_learning\Final_project\INFERENCE_DATA\Left cerebellar hemisphere\Case 3_DWI.jpg" 	
			--labels "C:\Users\ashwin\Documents\CIE_Deep_learning\Final_project\Acute_infarct.txt"

EXPLANATION:
	1. train.py code is used to train the model on the datasets provided in the FILTERED_DATA folder. We have obtained 38 unique classes after data optimisations. 
		We have split the 50 images into 45 and 5 to train and test, used the image datagenerator to only 45 training images and look for prediction on the 5
		test images.

	2. predictions.py is used to check for the accuracy of the trained images and test images present in the FILTERED_DATA folder,
		and the accuracy of 36 inference images present in the INFERENCE_DATA.

	3. get_IR_from_h5.py  code is used to convert the .h5 file to a .pb file (frozen model) which is given as an input to the model optimiser,
		The model optimizer code has been implemented within the get_IR_from_h5.py file itself.
		Half way down the completion of the code, it will ask for the path to the model optimizer in your system where you will have to enter its path and hit enter.
		After that is done, a .xml , .bin and .mappings file will successfully be created.
	
	4. The final application code first requires the openvino environments to be initialised. Once that is done the model takes 3 parameters as input
		i. '.xml' file (IR represntation model)
		ii. inference image
		iii. labels file.
		and using these three files it predicts the output class.