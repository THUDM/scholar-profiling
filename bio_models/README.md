This folder provides three profiling methods for extracting more than 10 attributes from long texts of scholars' profiles.

### Implemented Methods
- CNN [1]
- EGPointer [2]
- UIE [3]  

Required pacakges: see `requirements.txt`.

### Data
- Dataset used for CNN and EGPointer is located in `bio_models/en_bio`
- Dataset used for UIE is located in `bio_models/UIE/data/text2spotasoc/en_bio/en_bio`

### How to run
The running guide is in the README.md in the folder of each method.

### Results
|       | F1 |
|-------|-------|
| CNN   | 0.4391 |
| EGpointer | 0.4187 |
| UIE   | 0.3715 |

### References
[1] Yan, Hang, Yu Sun, Xiaonan Li, and Xipeng Qiu. "An Embarrassingly Easy but Strong Baseline for Nested Named Entity Recognition." arXiv preprint arXiv:2208.04534 (2022).  
[2] Su, Jianlin, Ahmed Murtadha, Shengfeng Pan, Jing Hou, Jun Sun, Wanwei Huang, Bo Wen, and Yunfeng Liu. "Global Pointer: Novel Efficient Span-based Approach for Named Entity Recognition." arXiv preprint arXiv:2208.03054 (2022).  
[3] Lu, Yaojie, Qing Liu, Dai Dai, Xinyan Xiao, Hongyu Lin, Xianpei Han, Le Sun, and Hua Wu. "Unified Structure Generation for Universal Information Extraction." In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5755-5772. 2022.
  
