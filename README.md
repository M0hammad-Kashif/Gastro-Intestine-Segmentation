# Gastro-Intestinal-Image-Segmentation


> In 2019, an estimated 5 million people were diagnosed with a cancer of the gastro-intestinal tract worldwide. Of these patients, about half are eligible for radiation therapy, usually delivered over 10-15 minutes a day for 1-6 weeks. Radiation oncologists try to deliver high doses of radiation using X-ray beams pointed to tumors while avoiding the stomach and intestines. With newer technology such as integrated magnetic resonance imaging and linear accelerator systems, also known as MR-Linacs, oncologists are able to visualize the daily position of the tumor and intestines, which can vary day to day. In these scans, radiation oncologists must manually outline the position of the stomach and intestines in order to adjust the direction of the x-ray beams to increase the dose delivery to the tumor and avoid the stomach and intestines. This is a time-consuming and labor intensive process that can prolong treatments from 15 minutes a day to an hour a day, which can be difficult for patients to tolerate—unless deep learning could help automate the segmentation process. A method to segment the stomach and intestines would make treatments much faster and would allow more patients to get more effective treatment.
> 
> The MRI scans are from actual cancer patients who had 1-5 MRI scans on separate days during their radiation treatment
> 
> The UW-Madison Carbone Cancer Center is a pioneer in MR-Linac based radiotherapy, and has treated patients with MRI guided radiotherapy based on their daily anatomy since 2015. UW-Madison has generously agreed to support this project which provides anonymized MRIs of patients treated at the UW-Madison Carbone Cancer Center. The University of Wisconsin-Madison is a public land-grant research university in Madison, Wisconsin. The Wisconsin Idea is the university's pledge to the state, the nation, and the world that their endeavors will benefit all citizens.

# Architecture Used

A Nested U-Net Architecture i.e. Unet++ [https://doi.org/10.48550/arXiv.1807.10165] with Efficientnet b-7 [https://doi.org/10.48550/arXiv.1905.11946] as a backbone architecture.

Model link :
[Unet++](https://drive.google.com/file/d/1umSTGMgIzYoIonmizi53v5fXtI0EAGMg/view?usp=sharing)

# Gastro-Intestinal-Images

![Sample Image](sampleImages/sampleTraining.png)

![Model Output](sampleImages/giOutput.png)

# Authors

- [Mohammad-Kashif](https://github.com/M0hammad-Kashif/)

- [Haider Zama](https://github.com/haiderzm)