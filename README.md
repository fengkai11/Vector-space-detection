# Vector-space-detection
In [AI Day](https://www.youtube.com/watch?v=j0z4FweCy4M), Andrej Karpathy, the Sr. Director of Tesla AI,introduced the FSD perception algorithm and showed more details of
the architectue of network used in FSD.This is the detection part to verify my deduction about the details of implement.The key words shows as below：  
- Resne50  
- BIFnet  
- Feature
- transformer
- image space  
- vector space
- yolo style
- bird veiw  

The crucial problem is how convert image space to vector space, Andrej Karpathy has introduced the technology of transformer,but no details of the pipline. In this repository 
I am trying to verify this part.  

The blogs about AI day can be refered:  
[1.Deep Understanding Tesla FSD,Jason Zhang](https://saneryee-studio.medium.com/deep-understanding-tesla-fsd-part-1-hydranet-1b46106d57)    
[2.Tesla AutoPilot 纯视觉方案解析](https://zhuanlan.zhihu.com/p/404916271)

**TODO list**  
- [ ] 1. build network;
- [ ] 2. prepare kitti dataset and dataloader;
- [ ] 3. train the newwork;
- [ ] 4. writre the tracker part;

