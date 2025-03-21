# Solution for Team AimF-SR: Mixture of Attention for Efficient Super Resolution -- [NTIRE 2025 Challenge on Efficient Super-Resolution](https://cvlai.net/ntire/2025/)

<div align=center>
<img src="https://github.com/Amazingren/NTIRE2025_ESR/blob/main/figs/logo.png" width="400px"/> 
</div>

## News
- :t-rex: February 8th, 2025: Our Challenge Repo. is ready!


## The Environments

The evaluation environments adopted by us is recorded in the `requirements.txt`. After you built your own basic Python (Python = 3.9 in our setting) setup via either *virtual environment* or *anaconda*, please try to keep similar to it via:


## Install

- Step 1: Make anaconda environment or virutal environment
Create a conda enviroment:
````
ENV_NAME="ntire25"
conda create -n $ENV_NAME python=3.9
conda activate $ENV_NAME
````

- Step2: install Pytorch compatible to your GPU (in this case, we follow the environment setting for NTIRE 2025 ESR):
````  
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
````

- Step3: install other libs via:
  ````
  pip install -r requirements.txt
  ````

The environment setting is kept as similar with [NTIRE2025 ESR](https://github.com/Amazingren/NTIRE2025_ESR)


## The Validation datasets
After downloaded all the necessary validate dataset ([DIV2K_LSDIR_valid_LR](https://drive.google.com/file/d/1YUDrjUSMhhdx1s-O0I1qPa_HjW-S34Yj/view?usp=sharing) and [DIV2K_LSDIR_valid_HR](https://drive.google.com/file/d/1z1UtfewPatuPVTeAAzeTjhEGk4dg2i8v/view?usp=sharing)), please organize them as follows:


## How to add your model to this baseline?

1. Register your team in the [Google Spreadsheet](https://docs.google.com/spreadsheets/d/11JuxcS78C6Gxc8B436L4Zk4_m5soHaTcw3cnF8h5ctE/edit?usp=sharing) and get your team ID.
2. Put your the code of your model in `./models/[Your_Team_ID]_[Your_Model_Name].py`
   - Please add **only one** file in the folder `./models`. **Please do not add other submodules**.
   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02 
3. Put the pretrained model in `./model_zoo/[Your_Team_ID]_[Your_Model_Name].[pth or pt or ckpt]`
   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02  
4. Add your model to the model loader `./test_demo/select_model` as follows:
    ```python
        elif model_id == [Your_Team_ID]:
            # define your model and load the checkpoint
    ```
   - Note: Please set the correct data_range, either 255.0 or 1.0
5. Send us the command to download your code, e.g, 
   - `git clone [Your repository link]`
   - We will do the following steps to add your code and model checkpoint to the repository.
   

## How to calculate the number of parameters, FLOPs, and activations

```python
    from utils.model_summary import get_model_flops, get_model_activation
    from models.team00_EFDN import EFDN
    from fvcore.nn import FlopCountAnalysis

    model = EFDN()
    
    input_dim = (3, 256, 256)  # set the input dimension
    activations, num_conv = get_model_activation(model, input_dim)
    activations = activations / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
    print("{:>16s} : {:<d}".format("#Conv2d", num_conv))

    # The FLOPs calculation in previous NTIRE_ESR Challenge
    # flops = get_model_flops(model, input_dim, False)
    # flops = flops / 10 ** 9
    # print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

    # fvcore is used in NTIRE2025_ESR for FLOPs calculation
    input_fake = torch.rand(1, 3, 256, 256).to(device)
    flops = FlopCountAnalysis(model, input_fake).total()
    flops = flops/10**9
    print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    num_parameters = num_parameters / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
```



## References
If you feel this codebase and the report paper is useful for you, please cite our challenge report:
```
@inproceedings{ren2024ninth,
  title={The ninth NTIRE 2024 efficient super-resolution challenge report},
  author={Ren, Bin and Li, Yawei and Mehta, Nancy and Timofte, Radu and Yu, Hongyuan and Wan, Cheng and Hong, Yuxin and Han, Bingnan and Wu, Zhuoyuan and Zou, Yajun and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6595--6631},
  year={2024}
}
```


## License and Acknowledgement
This code repository is release under [MIT License](LICENSE). 
