import torch
import benepar

if __name__ == '__main__':
    classifer = torch.load('models/span_classifier_model.pth')
    intensity_classifer = torch.load('models/span_intensity_cls_model.pth')
    parser = benepar.Parser("benepar_en3")
    
    classifer.eval()
    intensity_classifer.eval()

    pass