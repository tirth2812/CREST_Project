\# CREST YOLO26 Vision System



This folder contains the final organized version of the CREST YOLO26 vision system.



The project is organized stage-by-stage so the complete development flow can be understood, reproduced, and reused later.



\---



\# Project Structure



\## 01\_Pallet\_Annotation



Contains the pallet annotation tool and the pallet annotation data used to prepare the pallet localization dataset.



Main contents:



\- CODE

&#x20; - pallet\_annotator.py



\- DATA

&#x20; - pallet\_annotations.json

&#x20; - images



\---



\## 02\_YOLO\_Pallet\_Detection



Contains the pallet localization model training, evaluation, dataset, model weights, training results, and visual test results.



Main contents:



\- CODE

&#x20; - train\_yolo26\_pallet\_localizer.py

&#x20; - evaluate\_yolo\_pallet\_localizer\_test.py



\- CONFIG\_AND\_RESULTS

&#x20; - pallet\_annotations.json

&#x20; - split\_used\_for\_yolo.json

&#x20; - yolo26\_localizer\_config.json

&#x20; - yolo26\_test\_metrics.json

&#x20; - training\_output



\- DATASET

&#x20; - yolo\_dataset

&#x20;   - train

&#x20;   - val

&#x20;   - test



\- MODEL

&#x20; - pallet\_yolo26n\_best.pt



\- VISUAL\_TEST\_RESULTS



\---



\## 03\_Empty\_Object\_Classifier



Contains the classifier used to determine whether a detected pallet is empty or contains an object.



Main contents:



\- CODE

&#x20; - train\_yolo26\_empty\_object\_classifier.py



\- DATASET

&#x20; - prepared\_yolo26

&#x20;   - train

&#x20;   - val

&#x20;   - test



\- MODEL

&#x20; - empty\_object\_yolo26n\_cls\_best.pt



\- RESULTS

&#x20; - classifier evaluation results

&#x20; - held-out predictions

&#x20; - training results



\---



\## 04\_Live\_Empty\_Object\_System



Contains the live vision implementation for classifying pallets as:



\- EMPTY

\- OBJECT\_PRESENT



Main code:



C:\\Do\_Not\_Delete\_PLC\\FINAL\\04\_Live\_Empty\_Object\_System\\CODE\\live\_empty\_object\_classification.py



\---



\## 05\_Dynamic\_Pallet\_Detection



Contains the dynamic pallet detection and tracking system.



The tracker maintains pallet IDs while pallets move around the conveyor.



Main contents:



\- CODE

&#x20; - dynamic\_live\_pallet\_detection.py



\- CONFIG

&#x20; - conveyor\_boundary.json



\---



\## 06\_Live\_Empty\_Object\_Detection\_Final



Contains the combined live pallet detection and EMPTY / OBJECT\_PRESENT classification system.



Main code:



C:\\Do\_Not\_Delete\_PLC\\FINAL\\06\_Live\_Empty\_Object\_Detection\_Final\\CODE\\live\_object\_detection\_on\_pallet.py



Configuration:



C:\\Do\_Not\_Delete\_PLC\\FINAL\\06\_Live\_Empty\_Object\_Detection\_Final\\CONFIG\\conveyor\_boundary.json



\---



\# 07\_Final\_Empty\_Accepted\_Rejected\_System



This is the final CREST runtime system.



The final pipeline performs:



1\. Pallet detection

2\. Dynamic pallet tracking

3\. EMPTY vs OBJECT\_PRESENT classification

4\. X vs NO\_X symbol classification

5\. Final pallet status decision

6\. State preservation during temporary occlusions

7\. State transfer when tracker IDs change

8\. Live display of pallet status



\---



\## Final Decision Logic



The final status logic is:



EMPTY pallet:



EMPTY



Object present with X symbol:



REJECTED



Object present with NO\_X / tick symbol:



ACCEPTED



Therefore:



EMPTY -> EMPTY



OBJECT\_PRESENT + X -> REJECTED



OBJECT\_PRESENT + NO\_X -> ACCEPTED



\---



\## Final Runtime Code



Main runtime:



C:\\Do\_Not\_Delete\_PLC\\FINAL\\07\_Final\_Empty\_Accepted\_Rejected\_System\\CODE\\live\_final\_crest\_yolo26.py



Dynamic pallet tracker:



C:\\Do\_Not\_Delete\_PLC\\FINAL\\07\_Final\_Empty\_Accepted\_Rejected\_System\\CODE\\dynamic\_live\_pallet\_detection.py



\---



\## Final Models



Pallet localization model:



C:\\Do\_Not\_Delete\_PLC\\FINAL\\07\_Final\_Empty\_Accepted\_Rejected\_System\\MODELS\\pallet\_yolo26n\_best.pt



EMPTY / OBJECT\_PRESENT classifier:



C:\\Do\_Not\_Delete\_PLC\\FINAL\\07\_Final\_Empty\_Accepted\_Rejected\_System\\MODELS\\empty\_object\_yolo26n\_cls\_best.pt



X / NO\_X classifier:



C:\\Do\_Not\_Delete\_PLC\\FINAL\\07\_Final\_Empty\_Accepted\_Rejected\_System\\MODELS\\x\_no\_x\_yolo26n\_medium\_focus\_final\_hard\_best.pt



\---



\## Conveyor Configuration



C:\\Do\_Not\_Delete\_PLC\\FINAL\\07\_Final\_Empty\_Accepted\_Rejected\_System\\CONFIG\\conveyor\_boundary.json



This file contains the conveyor boundary information used by the live pallet tracking system.



\---



\## Final Results



Final testing and validation evidence is stored under:



C:\\Do\_Not\_Delete\_PLC\\FINAL\\07\_Final\_Empty\_Accepted\_Rejected\_System\\RESULTS



Important result folders include:



\- External\_Validation

\- Final\_Hard\_Data\_Split

\- Final\_Live\_Test\_Evidence

\- Final\_Model\_Training



\---



\# Final Runtime Environment



Python environment:



C:\\VLM\_Finetune\\vlm\_env



Main libraries:



\- Python

\- OpenCV

\- PyTorch

\- Ultralytics

\- NumPy



GPU used for development and testing:



NVIDIA RTX 5090



\---



\# Run Final CREST YOLO26 System



Open Command Prompt.



Run:



cd /d C:\\VLM\_Finetune



Activate the environment:



call vlm\_env\\Scripts\\activate.bat



Run the final system:



python "C:\\Do\_Not\_Delete\_PLC\\FINAL\\07\_Final\_Empty\_Accepted\_Rejected\_System\\CODE\\live\_final\_crest\_yolo26.py"



\---



\# Final Runtime Files Required



The final live system requires:



1\. live\_final\_crest\_yolo26.py



2\. dynamic\_live\_pallet\_detection.py



3\. conveyor\_boundary.json



4\. pallet\_yolo26n\_best.pt



5\. empty\_object\_yolo26n\_cls\_best.pt



6\. x\_no\_x\_yolo26n\_medium\_focus\_final\_hard\_best.pt



These files are already organized inside:



C:\\Do\_Not\_Delete\_PLC\\FINAL\\07\_Final\_Empty\_Accepted\_Rejected\_System



\---



\# Final Runtime Code Verification



The final runtime code was copied from the latest tested Phase 27 implementation.



SHA256:



d43e39782cf2eda35e441e07339e109da8e995e2cb5e57bbf1f8d01bad2919f5



This hash was verified to match the tested development version.



\---



\# Development Folder



The active development and Git repository is:



C:\\Do\_Not\_Delete\_PLC\\CREST\_YOLO26



Development changes should be made and tested there first.



After a change is fully tested and confirmed working, the corresponding finalized files can be copied into:



C:\\Do\_Not\_Delete\_PLC\\FINAL



The FINAL folder should remain the clean transfer and backup version of the project.



\---



\# 08\_Reusable\_Datasets



This folder contains cleaned reusable pallet image data preserved for future machine learning and computer vision projects.



It is separate from the final CREST runtime and is not required to run Stage 07.



Dataset location:



C:\\Do\_Not\_Delete\_PLC\\FINAL\\08\_Reusable\_Datasets\\Clean\_Pallet\_Dataset



\---



\## Dataset Structure



Clean\_Pallet\_Dataset



train

\- ACCEPTED

\- EMPTY

\- REJECTED



val

\- ACCEPTED

\- EMPTY

\- REJECTED



Full structure:



C:\\Do\_Not\_Delete\_PLC\\FINAL\\08\_Reusable\_Datasets\\Clean\_Pallet\_Dataset

|

+---train

|   +---ACCEPTED

|   +---EMPTY

|   \\---REJECTED

|

\\---val

&#x20;   +---ACCEPTED

&#x20;   +---EMPTY

&#x20;   \\---REJECTED



\---



\## Training Dataset Counts



ACCEPTED:



60 images



EMPTY:



95 images



REJECTED:



60 images



Training total:



215 images



\---



\## Validation Dataset Counts



ACCEPTED:



18 images



EMPTY:



25 images



REJECTED:



18 images



Validation total:



61 images



\---



\## Complete Reusable Dataset



ACCEPTED:



78 images



EMPTY:



120 images



REJECTED:



78 images



Total:



276 images



\---



\## Dataset Sources



The EMPTY images were taken from the cleaned final real-live EMPTY vs OBJECT\_PRESENT dataset:



C:\\Do\_Not\_Delete\_PLC\\original images\\06\_Empty\_vs\_Object\_Classifier\\04\_Improved\_Classifier\_Training\\final\_real\_live\_classifier\_dataset



The ACCEPTED and REJECTED images were taken from the balanced full-pallet dataset:



C:\\Do\_Not\_Delete\_PLC\\original images\\15\_Accepted\_vs\_Rejected\_Classifier\\balanced\_full\_pallet\_dataset



The cleaned datasets were selected instead of the earlier raw close-pallet folders so problematic raw images are not carried forward into future model development.



\---



\## Purpose of 08\_Reusable\_Datasets



This dataset is preserved so future projects can reuse clean labeled pallet images without having to search through the original development folders again.



Possible future uses include:



\- image classification

\- machine learning experiments

\- transfer learning

\- ACCEPTED / REJECTED classification

\- EMPTY / object classification

\- three-class pallet classification

\- computer vision experiments



This dataset is not part of the Stage 07 runtime and can be reused independently.



\---



\# Final Package Location



C:\\Do\_Not\_Delete\_PLC\\FINAL



The FINAL folder now contains:



01\_Pallet\_Annotation



02\_YOLO\_Pallet\_Detection



03\_Empty\_Object\_Classifier



04\_Live\_Empty\_Object\_System



05\_Dynamic\_Pallet\_Detection



06\_Live\_Empty\_Object\_Detection\_Final



07\_Final\_Empty\_Accepted\_Rejected\_System



08\_Reusable\_Datasets



README.md



This folder is the organized final CREST YOLO26 project package intended for backup, transfer, reference, and future reuse.

