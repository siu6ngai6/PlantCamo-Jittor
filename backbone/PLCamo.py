import jittor as jt
from jittor import nn

from lib.pvtv2 import pvt_v2_b2

class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv(64, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.gauss_(0, 0.01)

    def execute(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out

    def initialize(self):
        pass


class Att(nn.Module):
    def __init__(self, channels=64, r=4):
        super(Att, self).__init__()
        out_channels = int(channels // r)
        # local_att
        self.local_att = nn.Sequential(
            nn.Conv(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm(out_channels),
            nn.ReLU(),
            nn.Conv(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm(channels)
        )

        # global_att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm(out_channels),
            nn.ReLU(),
            nn.Conv(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm(channels)
        )

        self.sig = nn.Sigmoid()

    def execute(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sig(xlg)

        return wei

    def initialize(self):
        # weight_init(self)
        pass




class MyNet(nn.Module):
    def __init__(self, channel = 32) :
        super(MyNet,self).__init__()

        #backbone 的加载
        self.backbone = pvt_v2_b2()
        path = './pretrained_pvt/pvt_v2_b2.pth'
        save_model = jt.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)


        self.cr4 = nn.Sequential(nn.Conv(512,64,1), nn.BatchNorm(64), nn.ReLU())
        self.cr3 = nn.Sequential(nn.Conv(320,64,1), nn.BatchNorm(64), nn.ReLU())
        self.cr2 = nn.Sequential(nn.Conv(128,64,1), nn.BatchNorm(64), nn.ReLU())
        self.cr1 = nn.Sequential(nn.Conv(64,64,1), nn.BatchNorm(64), nn.ReLU())

        self.conv0 = nn.Sequential(self._make_pred_layer(Classifier_Module, [1, 6, 8, 12], [1, 6, 8, 12], 64),nn.BatchNorm(64), nn.ReLU())
        self.conv1 = nn.Sequential(self._make_pred_layer(Classifier_Module, [1, 6, 8, 12], [1, 6, 8, 12], 64),nn.BatchNorm(64), nn.ReLU())
        self.conv2 = nn.Sequential(self._make_pred_layer(Classifier_Module, [1, 6, 8, 12], [1, 6, 8, 12], 64),nn.BatchNorm(64), nn.ReLU())
        self.conv3 = nn.Sequential(self._make_pred_layer(Classifier_Module, [1, 6, 8, 12], [1, 6, 8, 12], 64),nn.BatchNorm(64), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv(64,64,3,1,1,1), nn.BatchNorm(64), nn.ReLU())
        
        self.cbr1 = nn.Sequential(nn.Conv(64,64,3,1,1,1), nn.BatchNorm(64), nn.ReLU())
        self.cbr2 = nn.Sequential(nn.Conv(64,64,3,1,1,1), nn.BatchNorm(64), nn.ReLU())
        self.cbr3 = nn.Sequential(nn.Conv(64,64,3,1,1,1), nn.BatchNorm(64), nn.ReLU())
        self.cbr4 = nn.Sequential(nn.Conv(64,64,3,1,1,1), nn.BatchNorm(64), nn.ReLU())
        

        self.Att0 = Att()
        self.Att1 = Att()
        self.Att2 = Att()
        self.Att3 = Att()


        self.map = nn.Conv(64, 1, 7, 1, 3)

        # self.fm1 = Focus(64,64)
        # self.fm2 = Focus(64,64)
        # self.fm3 = Focus(64,64)
        # self.fm4 = Focus(64,64)

        self.out_map = nn.Conv(64, 1, 7, 1, 3)


        
    def _make_pred_layer(self, block, dilation_series, padding_series, num_classes):
        return block(dilation_series, padding_series, num_classes)

    def execute(self, x):
        """
            x : 3  * 448 * 448
            x1: 64 * 176 * 176
            x2: 128* 88  * 88
            x3: 320* 44  * 44
            x4: 512* 22  * 22

        """

        #提取特征
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        x_4 = self.cr4(x4)  #64
        x_3 = self.cr3(x3)  #64
        x_2 = self.cr2(x2)  #64
        x_1 = self.cr1(x1)  #64

        

        

############################      Iterate  Block  ##################
        stage_loss1=list()
        stage_loss2=list()
        FeedBack_feature=None
        for iter in range(2):
            if FeedBack_feature is None:
                f_1 =x_1     #64
            else:
                f_1 =x_1 + FeedBack_feature    #
                
            f_2 = x_2           #64
            f_3 = x_3           #64
            f_4 = x_4           #64

            gf0 = f_1
            gf0 = self.conv0(f_1)
            gf0 = gf0*self.Att0(gf0)
            gf0 = nn.interpolate(gf0, size=x_2.size()[2:], mode='bilinear')

            # gf1 = gf0+f_2
            gf1 = self.conv1(gf0+f_2)
            gf1 = gf1*self.Att1(gf1)
            gf1 = nn.interpolate(gf1, size=x_3.size()[2:], mode='bilinear')

            # gf2 =gf1+f_3
            gf2 = self.conv2(gf1+f_3)
            gf2 = gf2*self.Att2(gf2)
            gf2 = nn.interpolate(gf2, size=x_4.size()[2:], mode='bilinear')

            # gf3 =gf2+f_4
            gf3 = self.conv3(gf2+f_4)
            gf3 = gf3*self.Att3(gf3)
            gf3 = self.conv4(gf3)
            gf_pre = self.map(gf3)

            

            # rf4,rf4_map = self.fm4(f_4,gf3,gf_pre,upsample = False)
            # rf3,rf3_map = self.fm3(f_3,rf4,rf4_map)
            # rf2,rf2_map = self.fm2(f_2,rf3,rf3_map)
            # rf1,rf1_map = self.fm1(f_1,rf2,rf2_map)
            rf4 = f_4 + gf3
            rf4 = self.cbr1(rf4)
            rf4 = nn.interpolate(rf4, size=x_3.size()[2:], mode='bilinear')
            rf3 = rf4 + f_3
            rf3 = self.cbr2(rf3)
            rf3 = nn.interpolate(rf3, size=x_2.size()[2:], mode='bilinear')
            rf2 = rf3 + f_2
            rf2 = self.cbr3(rf2)
            rf2 = nn.interpolate(rf2, size=x_1.size()[2:], mode='bilinear')
            rf1 = rf2 + f_1
            rf1 = self.cbr4(rf1)
            rf1_map = self.out_map(rf1)

            FeedBack_feature = rf1

            gf_pre = nn.interpolate(gf_pre, size=x.size()[2:], mode='bilinear')
            rf1_map= nn.interpolate(rf1_map, size=x.size()[2:], mode='bilinear')
            stage_loss1.append(gf_pre)
            stage_loss2.append(rf1_map)
           

        
        return stage_loss1,stage_loss2

