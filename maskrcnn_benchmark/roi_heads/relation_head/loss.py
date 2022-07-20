# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import numpy.random as npr

from maskrcnn_benchmark.layers import smooth_l1_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat


class RelationLoss_deng(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
        self,
        attri_on,
        num_attri_cat,
        max_num_attri,
        attribute_sampling,
        attribute_bgfg_ratio,
        use_label_smoothing,
        predicate_proportion,
        if_parent_model,
        USE_GT_BOX,
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        self.if_parent_model = if_parent_model
        self.attri_on = attri_on
        self.num_attri_cat = num_attri_cat
        self.max_num_attri = max_num_attri
        self.attribute_sampling = attribute_sampling
        self.attribute_bgfg_ratio = attribute_bgfg_ratio
        self.use_label_smoothing = use_label_smoothing
        self.pred_weight = (1.0 / torch.FloatTensor([0.5,] + predicate_proportion)).cuda()
        self.USE_GT_BOX = USE_GT_BOX

        if self.use_label_smoothing:
            self.criterion_loss = Label_Smoothing_Regression(e=0.01)
        else:
            # 用于四个阶段的effective num
            # self.criterion_loss0 = nn.CrossEntropyLoss(weight=torch.tensor([0.4669035425992956, 23.04961797159628, 0, 0, 0, 0, 0, 0, 16.526480564945707, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.5379842112009423, 14.28162893796638, 7.073231993848659, 0, 0, 0, 0, 0, 0, 8.65451558251361, 4.807590486310799, 1.6627411739115234, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4.0094246467283225, 0, 13.575230608673827]).cuda())
            # self.criterion_loss1 = nn.CrossEntropyLoss(weight=torch.tensor([49.333991119882846, 0, 0, 0, 0, 0, 438.27902734430694, 634.6053431721738, 0, 0, 0, 837.7240795800456, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 461.7049673440273, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 329.21053167630834, 0, 402.4698485261083, 917.5310824072942, 0, 377.2687719136511, 0, 0, 1586.9113768700145, 0, 0, 490.10462425755543, 0]).cuda())
            # self.criterion_loss2 = nn.CrossEntropyLoss(weight=torch.tensor([1.3626391608133395, 61.77079125734613, 1334.0056846812333, 1818.8508352079664, 164.61557276068348, 124.90532130216367, 306.4901395597556, 300.98008913197583, 50.83451458979665, 132.60771498665332, 175.20108696588733, 369.6829148433791, 165.97038576214655, 150.6063676589366, 128.39558181368437, 100000.0, 78.68589192207318, 2439.6891556758214, 1409.1225375935014, 106.95197718656645, 12.230007553206551, 49.01671945827162, 25.581564943172413, 146.45398262051197, 113.80408863818498, 76.15455369180575, 641.7028345460108, 9091.528489224118, 1266.495602311083, 27.662862869286624, 21.600328325389185, 6.530247408039948, 2128.3264705782567, 67.261086185069, 10000.613202941471, 134.37140569699417, 1539.1324767543072, 1923.7452207884544, 168.18528969908138, 10000.613202941471, 94.05302221800314, 188.29812681882606, 1786.3835254958628, 130.8899541248422, 142.72689565942005, 918.1064302469565, 385.2944860354449, 95.20027109238542, 13.624786655914551, 186.9006210341202, 47.456993808219366]).cuda())
            # self.criterion_loss3 = nn.CrossEntropyLoss(weight=torch.tensor([1.033837504085735, 60.398336180385094, 72.25380102560061, 86.50203293327607, 654.1084477080302, 813.52095578793, 249.8925454435387, 150.89267358456377, 41.086855887418224, 237.482894337381, 441.04347811905365, 422.45587701823257, 709.7332363970075, 709.7332363970075, 132.44303331145946, 1000.5118386231125, 317.97587576293347, 98.55653615301699, 102.35026090001827, 254.3230643018515, 8.903417558202182, 53.28878236152018, 19.47168179699047, 101.42545336878412, 241.47989821827224, 806.964473392422, 262.983092959305, 39.384147533409234, 61.79256541044361, 19.35039741833088, 18.25234674742833, 5.0524975470273, 226.24978077533711, 413.7381387614606, 32.5400279161331, 342.98116198345485, 123.97379215814134, 37.7630655542544, 167.4614974817113, 102.55808048088691, 71.13928383060926, 216.03345935810543, 233.07423914257083, 86.35442256302586, 595.7520867217534, 249.2722098428669, 257.5853451263085, 885.4681971007705, 12.040023597327265, 238.61130017370584, 36.119094235824086]).cuda())
            # 用于三个阶段的effective num
            # self.criterion_loss0 = nn.CrossEntropyLoss(weight=torch.tensor(
            #     [0.4669035425992956, 23.04961797159628, 0, 0, 0, 0, 0, 0, 16.526480564945707, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #      0, 0, 2.5379842112009423, 14.28162893796638, 7.073231993848659, 0, 0, 0, 0, 0, 0, 8.65451558251361,
            #      4.807590486310799, 1.6627411739115234, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #      4.0094246467283225, 0, 13.575230608673827]).cuda())
            # self.criterion_loss1 = nn.CrossEntropyLoss(weight=torch.tensor([34.89183531053541, 0, 0, 0, 3588.279796205593, 2239.326092054713, 425.7865447840164, 563.9843460612085, 0, 1466.5384402305754, 3465.149587921408, 817.4331769877967, 3242.721020962834, 2794.7755879052215, 4562.129744338858, 0, 1236.828550435729, 0, 0, 1207.7996410179528, 0, 0, 0, 400.7856364546382, 2794.7755879052215, 4364.536780651712, 0, 0, 0, 0, 0, 0, 0, 1267.3090067871383, 0, 2719.714625830004, 0, 0, 314.4719300277337, 0, 327.3171077148537, 830.4370083184199, 0, 345.5669953420286, 12515.27316879256, 0, 1387.144005346548, 1977.9398846306008, 0, 452.3860284852702, 0]).cuda())
            # self.criterion_loss2 = nn.CrossEntropyLoss(weight=torch.tensor(
            #     [1.033837504085735, 60.398336180385094, 72.25380102560061, 86.50203293327607, 654.1084477080302,
            #      813.52095578793, 249.8925454435387, 150.89267358456377, 41.086855887418224, 237.482894337381,
            #      441.04347811905365, 422.45587701823257, 709.7332363970075, 709.7332363970075, 132.44303331145946,
            #      1000.5118386231125, 317.97587576293347, 98.55653615301699, 102.35026090001827, 254.3230643018515,
            #      8.903417558202182, 53.28878236152018, 19.47168179699047, 101.42545336878412, 241.47989821827224,
            #      806.964473392422, 262.983092959305, 39.384147533409234, 61.79256541044361, 19.35039741833088,
            #      18.25234674742833, 5.0524975470273, 226.24978077533711, 413.7381387614606, 32.5400279161331,
            #      342.98116198345485, 123.97379215814134, 37.7630655542544, 167.4614974817113, 102.55808048088691,
            #      71.13928383060926, 216.03345935810543, 233.07423914257083, 86.35442256302586, 595.7520867217534,
            #      249.2722098428669, 257.5853451263085, 885.4681971007705, 12.040023597327265, 238.61130017370584,
            #      36.119094235824086]).cuda())

            # 用于测试没有reweighting的结果
            # self.criterion_loss0 = nn.CrossEntropyLoss(weight=torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]).float()).cuda()
            # self.criterion_loss1 = nn.CrossEntropyLoss(weight=torch.tensor([1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0]).float()).cuda()
            # self.criterion_loss2 = nn.CrossEntropyLoss(weight=torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).float()).cuda()

            # rebuttal一个阶段
            # 进行了sampling
            # self.criterion_loss0 = nn.CrossEntropyLoss(weight=torch.tensor([0.4669035425992956, 23.04961797159628, 0, 0, 0, 0, 0, 0, 16.526480564945707, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.5379842112009423, 14.28162893796638, 7.073231993848659, 0, 0, 0, 0, 0, 0, 8.65451558251361, 4.807590486310799, 1.6627411739115234, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4.0094246467283225, 0, 13.575230608673827]).cuda())
            # self.criterion_loss1 = nn.CrossEntropyLoss(weight=torch.tensor(
            #     [0.2119143981004199, 10.490681800596075, 75.29191226011655, 88.20964874992328, 166.49317232332493,
            #      129.13612441743288, 41.12124502500589, 48.48329042861239, 6.808263940253515, 105.25636860647573,
            #      144.19606679607304, 54.070523990483025, 167.32790053335395, 162.97027488449677, 71.2277046338677,
            #      1000.1029364615681, 73.74164745980372, 99.60639290588645, 108.79954489631507, 90.27524431160951,
            #      1.3822890729357777, 7.806812440054, 3.605418364638716, 22.043561143813672, 92.69650782326768,
            #      90.60165586616989, 265.35569976993366, 39.16652421813532, 65.89343263790552, 4.123511565200457,
            #      2.80559843779452, 0.8499962477434386, 234.29578194631543, 68.87974439207161, 32.78378743370276,
            #      116.2479014351596, 130.31219821121545, 37.41746319662442, 20.545664814315046, 102.14471813020235,
            #      15.55059079882634, 35.40230835402078, 240.4883530252302, 17.68194099140549, 139.76865764122473,
            #      257.17312806042145, 63.11593653056931, 100.20400460624, 1.9086155221776742, 19.159019929406565,
            #      7.020594021321701]
            # ).cuda())
            # 没有sampling、
            self.criterion_loss0 = nn.CrossEntropyLoss(weight=torch.tensor([0.4669035425992956, 23.04961797159628, 0, 0, 0, 0, 0, 0, 16.526480564945707, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.5379842112009423, 14.28162893796638, 7.073231993848659, 0, 0, 0, 0, 0, 0, 8.65451558251361, 4.807590486310799, 1.6627411739115234, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4.0094246467283225, 0, 13.575230608673827]).cuda())
            self.criterion_loss1 = nn.CrossEntropyLoss(weight=torch.tensor(
                [0.2486224691804972, 12.012798778863639, 380.3508769548616, 446.55122797252244, 202.96272683481956,
                 147.39845311230934, 47.5390803607065, 63.1749001458362, 7.788443013021899, 153.49727235437567,
                 195.81769619815674, 58.77423558449167, 206.30853272622377, 195.43548047695387, 142.5731976631588,
                 20000.098556451987, 89.7288760261907, 505.1730880596869, 581.5178364778455, 124.03882065712794,
                 1.5758111546610114, 8.833049427334313, 4.211064722228781, 26.383859687821676, 128.49278120456725,
                 97.58901396228484, 271.1255900229084, 781.372239203109, 377.4812342751968, 4.941427138837599,
                 3.176420652662165, 0.9763492484730911, 291.66804287748806, 78.43169895218664, 653.7171689749584,
                 156.12927567615074, 230.00799146103026, 746.390939318499, 22.311103875790128, 2040.9370100110002,
                 18.79757941598667, 40.187374670667765, 305.93323254313265, 21.256121988867843, 172.53680523065555,
                 346.1435447756276, 75.76613408645598, 110.37668849009923, 2.184025373242792, 20.37043367720789,
                 8.310458806535278]
            ).cuda())

            # 用于测试只有2个阶段的结果
            # self.criterion_loss0 = nn.CrossEntropyLoss(weight=torch.tensor(
            #     [0.4669035425992956, 23.04961797159628, 0, 0, 0, 0, 0, 0, 16.526480564945707, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #      0, 0, 2.5379842112009423, 14.28162893796638, 7.073231993848659, 0, 0, 0, 0, 0, 0, 8.65451558251361,
            #      4.807590486310799, 1.6627411739115234, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #      4.0094246467283225, 0, 13.575230608673827]).cuda())
            # self.criterion_loss1 = nn.CrossEntropyLoss(weight=torch.tensor([1.033837504085735, 60.398336180385094, 72.25380102560061, 86.50203293327607, 654.1084477080302,
            #  813.52095578793, 249.8925454435387, 150.89267358456377, 41.086855887418224, 237.482894337381,
            #  441.04347811905365, 422.45587701823257, 709.7332363970075, 709.7332363970075, 132.44303331145946,
            #  1000.5118386231125, 317.97587576293347, 98.55653615301699, 102.35026090001827, 254.3230643018515,
            #  8.903417558202182, 53.28878236152018, 19.47168179699047, 101.42545336878412, 241.47989821827224,
            #  806.964473392422, 262.983092959305, 39.384147533409234, 61.79256541044361, 19.35039741833088,
            #  18.25234674742833, 5.0524975470273, 226.24978077533711, 413.7381387614606, 32.5400279161331,
            #  342.98116198345485, 123.97379215814134, 37.7630655542544, 167.4614974817113, 102.55808048088691,
            #  71.13928383060926, 216.03345935810543, 233.07423914257083, 86.35442256302586, 595.7520867217534,
            #  249.2722098428669, 257.5853451263085, 885.4681971007705, 12.040023597327265, 238.61130017370584,
            #  36.119094235824086]).cuda())

            weight_sampling = torch.tensor(
                [0, 0.3030003422760021, 0.6925611558211656, 0.20985658803687995, 0.25171325851407894,
                 0.3223738538385887, 0.6532611759582221, 0.16746522392835234, 0.17001183174017584, 0.16735381633314222,
                 0.1493742162786178, 0.13707650789135747, 0.18270280960563354, 0.35201378534579253, 0.2646510756560587,
                 0.3629700861475516, 0.7235854106038326, 0.34451719399268205, 0.2864794230591251, 0.37491152253735965,
                 0.08172931072452504, 0.26549091836411803, 0.04272141025536979, 0.0895204728588353, 0.7067343126147694,
                 0.3413515792238631, 0.11454567921588568, 0.10110408989007458, 0.1530524059559256, 0.3102133963421149,
                 0.13948653120998644, 0.40164302378294586, 0.3406560053725582, 0.2360913831783778, 0.3681669756534939,
                 0.681718551120222, 0.39406573533207584, 0.11184199912885963, 0.13860384591435107, 1.3374176310246786,
                 0.10482858931938323, 0.10304273056024059, 0.36952360526073763, 0.2276380156769874, 0.1786781238097475,
                 0.1209009507541287, 0.6916046374627004, 0.532717733171526, 0.20217376401125012, 0.18310196671821946,
                 0.4282358477678262, 0.6073158124380467, 0.10940565108619114, 0.11341290295857627, 0.10182755796719839,
                 0.37491152253735965, 0.6069478344802046, 0.0886265664640903, 0.09048123496404648, 0.2792268417071339,
                 0.11697568250456972, 0.06145976170810374, 0.1850561473952489, 0.3875167985437833, 0.1244669804306662,
                 0.34750120517904465, 0.13178853031133786, 0.21986650747710976, 0.31518329589158683, 0.6490280953089725,
                 0.5809027692356499, 0.5390227120305154, 0.2518395177141335, 0.16328074846327556, 0.07079517717544129,
                 0.2312006410194927, 0.1706751448048652, 0.2939472474400919, 0.011678071895474793, 0.9492860680599607,
                 0.15173767930666168, 0.40570626035045365, 0.4517833658637209, 0.3644209402226492, 0.20241804385153844,
                 0.36735777545806425, 0.45589323021237316, 0.13015492135687218, 0.4762571979238274, 0.5293404881331744,
                 0.1562836001732177, 0.03651039936157613, 0.313509033418819, 0.3078340712853112, 0.2694095445871815,
                 0.10796665387397443, 0.3294673341970164, 0.09212105524896605, 0.24737272851122952, 0.08313822695885997,
                 0.3688440412577481, 0.4857211549150597, 0.2508956559094116, 1.0411021348613503, 0.315877915893822,
                 0.3302265259112458, 0.5200002857596315, 0.7122633917258976, 0.37212897941901185, 0.3481042305934525,
                 0.29446475497278074, 0.03875726832772927, 0.16248988463497321, 0.15927570583530157, 0.146463194746271,
                 0.05862713555449106, 0.5765565862107883, 0.17552903712982612, 0.3227887470793024, 0.412381482036906,
                 0.9430265373321344, 0.08848706225704892, 0.647768856562812, 0.8458280676656788, 0.08569975188548123,
                 0.21287948869479426, 0.049727002002136256, 0.11904610361333264, 0.3854313672261695, 0.857417551717675,
                 0.276686850304547, 0.5073621491956551, 0.9046786245258744, 0.33996326481817324, 0.13733807234281237,
                 0.062101440367483024, 0.042186934018813134, 0.19613837511126248, 0.2616848670769536,
                 0.17810862807837735, 0.1923115639566366, 0.9565424047020622, 0.5863427935807795, 0.554536358013393,
                 0.14716946975853834, 0.03852913064275777, 0.4140847954469973, 0.2250878544940257, 0.5133402848140579,
                 0.029469033469185155, 0.14553200969605035]
            ).cuda()
            weight_nosampling = torch.tensor(
                [0, 0.48023225117602597, 0.8739780282112859, 0.23792058676143268, 0.2871491921437364,
                 0.43882808929508504, 0.6926572549197465, 0.18617972749022174, 0.18812825442558248, 0.18624862015353819,
                 0.17347744815266009, 0.17362698103286284, 0.2213182448908124, 0.4350208648504306, 0.33450596543890326,
                 0.39791442064156146, 0.7628107943454623, 0.36651437650336927, 0.33054027352718196, 0.434643777778948,
                 0.0928597033119145, 0.363460803482503, 0.05231667928716921, 0.1102181145711336, 0.7639744500504276,
                 0.39587296946259687, 0.13632044764580012, 0.10885389995286082, 0.17317915650879978,
                 0.33174198175731295, 0.17627090009144786, 0.4316504866271863, 0.4064601934418395, 0.2528867558137204,
                 0.48138517364827726, 0.7818656566096853, 0.4290650217990765, 0.12169697983195314, 0.16859915520417032,
                 1.3447012636305655, 0.11094362100289533, 0.11569180275838693, 0.4642228245708046, 0.2518726059639088,
                 0.2080853374362066, 0.16043766934249426, 0.7468843424622915, 0.7018779964353852, 0.24759089754126137,
                 0.21697298285508135, 0.4625097296904799, 0.7496793538037947, 0.11646500374173889, 0.12552323183429864,
                 0.1280866155731219, 0.41298716266528906, 0.7300103025405761, 0.09809260088352716, 0.10167830716428634,
                 0.3453248516560714, 0.1322131186444834, 0.06976449179879347, 0.2079992867758986, 0.44744350834496066,
                 0.13694947134151697, 0.42961643551315504, 0.15036216107967246, 0.26551695247646373, 0.6457770649080365,
                 0.6921786650765153, 0.6615542929039102, 0.6428765351251173, 0.27716493550668997, 0.2017427882202766,
                 0.07846885657857929, 0.4642228245708046, 0.22543654391626555, 0.4257860795553448, 0.013361830326919545,
                 1.2186421261029001, 0.19804986117513737, 0.4674693673222218, 0.476353341720086, 0.4134979437227154,
                 0.22229674688747728, 0.6126109096781475, 0.49714025451904054, 0.14647439616321525, 0.538250288628997,
                 0.5678308240131346, 0.18417014957639047, 0.04110982239319927, 0.3629349424246874, 0.34121559252415,
                 0.2757951323237173, 0.1399119645742225, 0.4204312998970541, 0.10739766055579116, 0.33054027352718196,
                 0.1138032149261305, 0.49714025451904054, 0.6441164196149161, 0.3764146529861375, 1.1593639229035542,
                 0.4015783539419453, 0.39108853102513846, 0.6122366011030561, 0.9897351780266825, 0.46059756830838905,
                 0.36839836618858524, 0.3469964533226065, 0.0443327782444194, 0.18190735234787586, 0.18076428818209359,
                 0.18400184516146603, 0.08941152072224096, 0.6416414194233193, 0.20390982516861708, 0.352357237394812,
                 0.44944901538065907, 1.0758842484390205, 0.11990600005310821, 0.7668991938947767, 0.9858371723288265,
                 0.12067938787448611, 0.22950171598621033, 0.05567219219665972, 0.13758435367354455, 0.4176302008518947,
                 1.4170457876574463, 0.3531011370276589, 0.539408982255496, 0.9946512761726656, 0.39870522041991097,
                 0.15706224496255217, 0.07567578482607497, 0.05306774134730684, 0.25735571992834666, 0.2959499493914631,
                 0.20121828828278446, 0.28649388663800796, 1.024156922470687, 0.9130243116715893, 0.6291513022684664,
                 0.18323160352522022, 0.0472408801264514, 0.47186948845396737, 0.29335627451014096, 0.674924581603106,
                 0.033839043166190214, 0.16237660641986468]
            ).cuda()
            self.criterion_loss = nn.CrossEntropyLoss(
                
            )

            self.learn_previous_model_loss_L1 = nn.L1Loss()
            self.learn_previous_model_loss_MSE = nn.MSELoss()
            self.learn_previous_model_loss_cross = nn.CrossEntropyLoss()
            # 这个是用于计算distill_loss的时候，单纯只去模仿前面出现的类别
            # 这里注意第一个1是background,stage0只有10个，stage1为10+10，stage2是全部
            self.stage0_predicate_list = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]).cuda()
            self.stage1_predicate_list = torch.tensor([1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1]).cuda()
            self.stage2_predicate_list = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).cuda()

            # 用于测试只有2个阶段的结果
            # self.stage1_predicate_list = self.stage2_predicate_list
            
            
            
        # 用于distillation阶段在前面给的confidence
        self.confidence_distillation = None

    def become_parent(self):
        self.if_parent_model = True
        self.training = True

    def set_stage(self, stage):
        self.stage = stage

    def confidence_for_distillation(self, previous_recall, current_recall):
        tmp = [1]
        for pre, cur in zip(previous_recall, current_recall):
            if cur == 0:
                tmp.append(0)
            else:
                tmp.append(pre / cur)
        self.confidence_distillation = torch.tensor(tmp, dtype=torch.float).cuda()

    def set_temperature(self, temperature):
        self.distill_temperature = torch.tensor(temperature, dtype=torch.float).cuda()


    def cross_entropy_with_temperature(self, previous_output, current_output, temperature, confidence = None):
        # 第二种情况的蒸馏
        if confidence == None:
            # 在precls的时候，obj的标签就是gt，返回0
            if previous_output[0][0] == -1000:
                return torch.tensor(0., device=previous_output[0][0].device, dtype=torch.float)
            # 在sgcls和sgdet两种previous和precls的时候previous的关系loss
            softmax_function = nn.Softmax(dim=1)
            current_softmax = softmax_function(current_output/temperature)
            previous_softmax = softmax_function(previous_output/temperature)
            if self.stage == 1:
                return (torch.diagonal(-1.0 * current_softmax.log() @ (previous_softmax * self.stage0_predicate_list).t(), 0).float()).mean()
            if self.stage == 2:
                return (torch.diagonal(-1.0 * current_softmax.log() @ (previous_softmax * self.stage1_predicate_list).t(), 0).float()).mean()
            if self.stage == 3:
                return (torch.diagonal(-1.0 * current_softmax.log() @ (previous_softmax * self.stage2_predicate_list).t(), 0).float()).mean()
        # 第三种情况带有confidence的蒸馏
        else:
            # 在precls的时候，obj的标签就是gt，返回0
            if previous_output[0][0] == -1000:
                return torch.tensor(0., device=previous_output[0][0].device, dtype=torch.float)
            # 在sgcls和sgdet两种previous和precls的时候previous的关系loss
            softmax_function = nn.Softmax(dim=1)
            current_softmax = softmax_function(current_output / temperature)
            previous_softmax = softmax_function(previous_output / temperature)
            if self.stage == 1:
                return (torch.diagonal(-1.0 * current_softmax.log() @ (previous_softmax * self.stage0_predicate_list * confidence).t(), 0).float()).mean()
            if self.stage == 2:
                return (torch.diagonal(-1.0 * current_softmax.log() @ (previous_softmax * self.stage1_predicate_list * confidence).t(), 0).float()).mean()
            if self.stage == 3:
                return (torch.diagonal(-1.0 * current_softmax.log() @ (previous_softmax * self.stage2_predicate_list * confidence).t(), 0).float()).mean()

    def cosine_dis(self, previous_output, current_output):
        # 进行L2 Normalization
        for i in range(previous_output.shape[0]):
            previous_output[i] /= math.sqrt(torch.sum(previous_output[i] ** 2))
            current_output[i] /= math.sqrt(torch.sum(current_output[i] ** 2))
        # 计算cosine距离
        return (torch.diagonal(-1.0 * previous_output @ current_output.t(), 0).float()).mean()

    def calculate_param_distill_loss(self, delta_theta, fisher_matrix, importance_scores):
        return 0.5 * torch.sum((delta_theta + fisher_matrix) * (delta_theta)**2)

    def double_distill_loss(self, previous_logits, current_logtis):
        # 这里面的logits应该是cat之后的
        old_class_previous = previous_logits * self.stage0_predicate_list
        old_class_current = current_logtis * self.stage0_predicate_list
        new_class_previous = previous_logits * [1 if num == 0 else 0 for num in self.stage0_predicate_list]
        new_class_current = current_logtis * [1 if num == 0 else 0 for num in self.stage0_predicate_list]
        norm_old_class_previous = torch.tensor([line - torch.sum(line).tolist() / len(self.stage0_predicate_list) for line in old_class_previous])
        norm_old_class_current = torch.tensor([line - torch.sum(line).tolist() / len(self.stage0_predicate_list) for line in old_class_current])
        norm_new_class_previous = torch.tensor([line - torch.sum(line).tolist() / (51 - len(self.stage0_predicate_list)) for line in new_class_previous])
        norm_new_class_current = torch.tensor([line - torch.sum(line).tolist() / (51 - len(self.stage0_predicate_list)) for line in new_class_current])
        norm_old_class_previous = norm_old_class_previous[norm_old_class_previous == 0] = 1
        norm_old_class_current = norm_old_class_current[norm_old_class_current == 0] = 1
        norm_new_class_previous = norm_new_class_previous[norm_new_class_previous == 0] = 1
        norm_new_class_current = norm_new_class_current[norm_new_class_current == 0] = 1
        return learn_previous_model_loss_MSE(norm_old_class_previous * norm_new_class_previous, norm_old_class_current * norm_new_class_current)


    def __call__(self, proposals, rel_labels, relation_logits, refine_logits,
                 refine_obj_logits_from_previous=None, relation_logits_from_previous=None,
                 delta_theta=None, fisher_matrix=None, importance_scores=None):

        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.

        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])

        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """
        if self.attri_on:
            if isinstance(refine_logits[0], (list, tuple)):
                refine_obj_logits, refine_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attri_on = False
                refine_obj_logits = refine_logits
        else:
            refine_obj_logits = refine_logits

        batch_size = len(relation_logits)

        """
        if self.stage == 1:
            print("stage1中前面模型预测的和现在当前预测的relation个数差异对比")
            # print(relation_logits[0])
            # print(relation_logits_from_previous[0])
            print([relations.shape[0] for relations in relation_logits])
            print([relations.shape[0] for relations in relation_logits_from_previous])
            print([propos.shape[0] for propos in refine_obj_logits])
            print([propos.shape[0] for propos in refine_obj_logits_from_previous])
        """
        if relation_logits_from_previous != None:
            for_debug_relation_count = [relations.shape[0] for relations in relation_logits]
            for_debug_relation_count_previous = [relations.shape[0] for relations in relation_logits_from_previous]
        # relation_logits = cat(relation_logits, dim=0)
        # refine_obj_logits = cat(refine_obj_logits, dim=0)

        # PreCls和SGCl
        if self.USE_GT_BOX:
            # print("测试点1")
            relation_logits = cat(relation_logits, dim=0)
            refine_obj_logits = cat(refine_obj_logits, dim=0)


        # SGDet进行一些裁剪，对于预测长度相同的结果保留
        else:
            # 第0个阶段没有parent直接cat就好了
            if self.if_parent_model == True or self.stage == 0:
                # print("测试点2")
                relation_logits = cat(relation_logits, dim=0)
                refine_obj_logits = cat(refine_obj_logits, dim=0)
            # 第123阶段，有了parent就需要进行relation数目不同的剔除了
            else:
                # print("测试点3")
                copy_relation_logits = relation_logits
                copy_refine_obj_logits = refine_obj_logits
                relation_logits = cat(relation_logits, dim=0)
                refine_obj_logits = cat(refine_obj_logits, dim=0)



        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        rel_labels = cat(rel_labels, dim=0)
        """
        print(proposals[0].get_field("labels"))
        print(rel_labels[0])
        tensor([76, 130, 145, 137, 137, 78, 26], device='cuda:0')
        tensor([31, 31, 31, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0], device='cuda:0')

        这里loss的计算就是，每一行对应的一个relation或者object，这一行计算一个entropy
        然后本次放进来的batch种所有的relations和object计算一个平均作为crossEntropyLoss
        """
        if self.stage == 0:
            loss_relation = self.criterion_loss0(relation_logits, rel_labels.long())
            loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())
        elif self.stage ==1:
            loss_relation = self.criterion_loss1(relation_logits, rel_labels.long())
            loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())


            """如果loss是nan打印一下
            if torch.any(torch.isnan(loss_relation)):
                print(refine_obj_logits.shape)
                print(fg_labels.shape)
                print(refine_obj_logits[0])
                print(fg_labels[0])
            """
        elif self.stage ==2:
            loss_relation = self.criterion_loss2(relation_logits, rel_labels.long())
            loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())
        elif self.stage == 3:
            loss_relation = self.criterion_loss3(relation_logits, rel_labels.long())
            loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())
        else:
            loss_relation = self.criterion_loss(relation_logits, rel_labels.long())
            loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())
        # 如果是儿子就学习parent模型的结果计算和parent得到结果的差距
        # 这里是希望尽可能的一致
        lower_previous_loss = nn.Softmax(dim=1)

        """
        计算每一行的softmax并且验证计算softmax后每一行是不是和 = 1
        print(lower_previous_loss(relation_logits))
        sum = 0
        for i in range(len(lower_previous_loss(relation_logits)[0])):
            sum  += lower_previous_loss(relation_logits)[0][i]
        print(sum)
        """

        # 这里是在stage 1 2 3的时候计算previous_loss
        if self.if_parent_model == False and relation_logits_from_previous != None \
                and refine_obj_logits_from_previous != None: # 保证第一个stage没有学习的parent

            """
            print("学习上一个stage的知识")
            print(len(relation_logits_from_previous))
            print(relation_logits_from_previous[0].shape) # [torch.Size([30, 51]),...,...]
            print(relation_logits_from_previous[0])
            print(relation_logits.shape) # torch.Size([3126, 51])前面的数字是一个batch的所有累加
            print(relation_logits[0])

            print(len(refine_obj_logits_from_previous))
            print(refine_obj_logits_from_previous[0].shape)
            print(refine_obj_logits_from_previous[0])
            """

            # PreCls和SGCl
            if self.USE_GT_BOX:
                relation_logits_from_previous = cat(relation_logits_from_previous, dim=0)
                refine_obj_logits_from_previous = cat(refine_obj_logits_from_previous, dim=0)
            # SGDet进行一些裁剪，对于预测长度相同的结果保留
            else:
                # 阶段0没有parent直接cat
                if self.stage == 0:
                    relation_logits_from_previous = cat(relation_logits_from_previous, dim=0)
                    refine_obj_logits_from_previous = cat(refine_obj_logits_from_previous, dim=0)
                # 如果relation的和previous的一样直接cat
                else:

                    if relation_logits.shape == cat(relation_logits_from_previous, dim=0).shape:
                        relation_logits_from_previous = cat(relation_logits_from_previous, dim=0)
                    # 对于relation预测数量不统一的筛选
                    else:
                        prune_relation_logits = torch.tensor([]).cuda()
                        prune_relation_logits_from_previous = torch.tensor([]).cuda()
                        for i in range(len(copy_relation_logits)):
                            if copy_relation_logits[i].shape[0] == relation_logits_from_previous[i].shape[0]:
                                prune_relation_logits = cat((prune_relation_logits, copy_relation_logits[i]))
                                prune_relation_logits_from_previous = cat((prune_relation_logits_from_previous,
                                                                           relation_logits_from_previous[i]))

                        relation_logits = prune_relation_logits
                        relation_logits_from_previous = prune_relation_logits_from_previous


                    # 如果obj和previous一样直接cat
                    if refine_obj_logits.shape == cat(refine_obj_logits_from_previous, dim=0).shape:
                        refine_obj_logits_from_previous = cat(refine_obj_logits_from_previous, dim=0)
                    # 对于obj预测数量不统一的筛选
                    else:
                        prune_refine_obj_logits = torch.tensor([]).cuda()
                        prune_refine_obj_logits_from_previous = torch.tensor([]).cuda()
                        for i in range(len(copy_refine_obj_logits)):
                            if copy_refine_obj_logits[i].shape[0] == refine_obj_logits_from_previous[i].shape[0]:
                                prune_refine_obj_logits = cat((prune_refine_obj_logits, copy_refine_obj_logits[i]))
                                prune_refine_obj_logits_from_previous = cat((prune_refine_obj_logits_from_previous,
                                                                             refine_obj_logits_from_previous[i]))

                        refine_obj_logits = prune_refine_obj_logits
                        refine_obj_logits_from_previous = prune_refine_obj_logits_from_previous
                    """
                    print("看下长度")
                    print(relation_logits.shape)
                    relation_logits = prune_relation_logits
                    relation_logits_from_previous = prune_relation_logits_from_previous
                    print(relation_logits.shape)
                    print(relation_logits_from_previous.shape)
                    """


            assert relation_logits.shape == relation_logits_from_previous.shape and \
                    refine_obj_logits.shape == refine_obj_logits_from_previous.shape

            # 在没有给GT boundingbox的时候，如果形状不相等返回loss = 0然后在最前面进行continue
            if relation_logits.shape != torch.Size([0]):
                # # 先过一个softmax进行归一然后再计算L2
                """第一个蒸馏的loss形式，这里所有类比的输出都进行L2计算"""
                # loss_previsous_study_relation = 0.5 * (self.learn_previous_model_loss_MSE(
                #     lower_previous_loss(relation_logits_from_previous / self.distill_temperature),
                #     lower_previous_loss(relation_logits / self.distill_temperature))) * len(relation_logits)

                """第二个蒸馏的loss形式(这里的 T = 2 是经验性的蒸馏参数)"""
                # loss_previsous_study_relation = 0.5 * self.cross_entropy_with_temperature(
                #     relation_logits_from_previous, relation_logits, 1)
                loss_previsous_study_relation = 0.5 * self.learn_previous_model_loss_L1(
                    lower_previous_loss(relation_logits_from_previous),
                    lower_previous_loss(relation_logits)
                ) * len(relation_logits)

                """第三个蒸馏的loss形式，crossEntropyLoss_like + confidence"""
                # 在每个stage的前1000个循环中现在的模型是没有recall评估的，所以直接赋成1
                # if self.confidence_distillation == None:
                #     if self.stage == 1:
                #         self.confidence_distillation = self.stage0_predicate_list
                #     if self.stage == 2:
                #         self.confidence_distillation = self.stage1_predicate_list
                #     if self.stage == 3:
                #         self.confidence_distillation = self.stage2_predicate_list
                # loss_previsous_study_relation = self.cross_entropy_with_temperature(
                #     relation_logits_from_previous, relation_logits, self.distill_temperature, self.confidence_distillation)

                """第四个蒸馏的loss形式，L2 + confidence"""
                # # 在每个stage的前1000个循环中现在的模型是没有recall评估的，所以直接赋成1
                # if self.confidence_distillation == None:
                #     if self.stage == 1:
                #         self.confidence_distillation = self.stage0_predicate_list
                #     if self.stage == 2:
                #         self.confidence_distillation = self.stage1_predicate_list
                #     if self.stage == 3:
                #         self.confidence_distillation = self.stage2_predicate_list
                #
                # """
                # 比较所有放进来的loss，不用confidence而对于old class进行筛选
                # 以及利用confidence得到的loss，三个的数量级比较
                # loss_previsous_study_relation = self.learn_previous_model_loss_MSE(
                #     lower_previous_loss(relation_logits_from_previous),
                #     lower_previous_loss(relation_logits)) * len(relation_logits)
                # print(loss_previsous_study_relation)
                # loss_previsous_study_relation = self.learn_previous_model_loss_MSE(
                #     lower_previous_loss(relation_logits_from_previous) * self.stage0_predicate_list,
                #     lower_previous_loss(relation_logits) * self.stage0_predicate_list) * len(relation_logits)
                # print(loss_previsous_study_relation)
                # """
                #
                # sqrt_confidence_distillation = self.confidence_distillation.clone()
                # for i in range(len(self.confidence_distillation)):
                #     sqrt_confidence_distillation[i] = sqrt_confidence_distillation[i] ** 0.5
                # # 这里的 T = 2 是蒸馏温度
                # loss_previsous_study_relation = self.learn_previous_model_loss_MSE(
                #     lower_previous_loss(relation_logits_from_previous / self.distill_temperature) * sqrt_confidence_distillation,
                #     lower_previous_loss(relation_logits / self.distill_temperature) * sqrt_confidence_distillation) * len(relation_logits)



            else:
                # 注意这里直接用obj的因为他必然不是0
                loss_previsous_study_relation = self.learn_previous_model_loss_MSE(refine_obj_logits, refine_obj_logits)
            if refine_obj_logits.shape != torch.Size([0]):
                """第一个蒸馏的loss形式，这里所有类比的输出都进行L2计算"""
                # 先过一个softmax进行归一然后再计算L2
                loss_previsous_study_obj = 0.5 * (self.learn_previous_model_loss_MSE(
                    lower_previous_loss(refine_obj_logits_from_previous),
                    lower_previous_loss(refine_obj_logits))) * len(refine_obj_logits)

                """第二个蒸馏的loss形式(这里的 T = 2 是经验性的蒸馏参数)"""
                # loss_previsous_study_obj = self.cross_entropy_with_temperature(
                #     refine_obj_logits_from_previous, refine_obj_logits, self.distill_temperature)

                """第三个蒸馏的loss形式，crossEntropyLoss_like + confidence，但是
                    这里没有办法使用confidence是因为这里的类别的obj_label
                """
                # loss_previsous_study_obj = self.cross_entropy_with_temperature(
                #     refine_obj_logits_from_previous, refine_obj_logits, self.distill_temperature)

                """第四个蒸馏的loss形式，L2 + confidence，但是
                    这里没有办法使用confidence是因为这里的类别的obj_label
                """
                # # 这里的 T = 2 是蒸馏温度
                # loss_previsous_study_obj = (self.learn_previous_model_loss_MSE(
                #     lower_previous_loss(refine_obj_logits_from_previous / self.distill_temperature),
                #     lower_previous_loss(refine_obj_logits / self.distill_temperature))) * len(refine_obj_logits)
            else:
                loss_previsous_study_obj = self.learn_previous_model_loss_MSE(refine_obj_logits, refine_obj_logits)

            # 直接计算L2
            # loss_previsous_study_relation = self.learn_previous_model_loss(
            #     relation_logits_from_previous, relation_logits)
            # loss_previsous_study_obj = self.learn_previous_model_loss(
            #     refine_obj_logits_from_previous, refine_obj_logits)

            """
            print("计算得到的从parent进行学习的loss")
            # 手写的L2
            mu_loss = 0.0
            for i in range(len(lower_previous_loss(relation_logits))):
                for j in range(51):
                    mu_loss += (lower_previous_loss(relation_logits)[i][j] - lower_previous_loss(relation_logits_from_previous)[i][j]) ** 2
            print(mu_loss/(51*len(lower_previous_loss(relation_logits))))
            # 函数算的L2
            print(loss_relation, loss_refine_obj, loss_previsous_study_obj, loss_previsous_study_relation)
            """
            return loss_relation, loss_refine_obj, loss_previsous_study_obj, \
                   loss_previsous_study_relation


        # The following code is used to calcaulate sampled attribute loss
        if self.attri_on:
            refine_att_logits = cat(refine_att_logits, dim=0)
            fg_attributes = cat([proposal.get_field("attributes") for proposal in proposals], dim=0)

            attribute_targets, fg_attri_idx = self.generate_attributes_target(fg_attributes)
            if float(fg_attri_idx.sum()) > 0:
                # have at least one bbox got fg attributes
                refine_att_logits = refine_att_logits[fg_attri_idx > 0]
                attribute_targets = attribute_targets[fg_attri_idx > 0]
            else:
                refine_att_logits = refine_att_logits[0].view(1, -1)
                attribute_targets = attribute_targets[0].view(1, -1)

            loss_refine_att = self.attribute_loss(refine_att_logits, attribute_targets,
                                             fg_bg_sample=self.attribute_sampling,
                                             bg_fg_ratio=self.attribute_bgfg_ratio)
            return loss_relation, (loss_refine_obj, loss_refine_att)
        # 这里是在stage 0的时候不需要计算previous_loss
        else:
            # 最后那个是stage0没有parent，直接计算一个loss_distill_param = 0
            return loss_relation, loss_refine_obj, \
                   self.learn_previous_model_loss_MSE(relation_logits, relation_logits), \
                   self.learn_previous_model_loss_MSE(refine_obj_logits, refine_obj_logits)

    def generate_attributes_target(self, attributes):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        assert self.max_num_attri == attributes.shape[1]
        device = attributes.device
        num_obj = attributes.shape[0]

        fg_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch.zeros((num_obj, self.num_attri_cat), device=device).float()

        for idx in torch.nonzero(fg_attri_idx).squeeze(1).tolist():
            for k in range(self.max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, fg_attri_idx

    def attribute_loss(self, logits, labels, fg_bg_sample=True, bg_fg_ratio=3):
        if fg_bg_sample:
            loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').view(-1)
            fg_loss = loss_matrix[labels.view(-1) > 0]
            bg_loss = loss_matrix[labels.view(-1) <= 0]

            num_fg = fg_loss.shape[0]
            # if there is no fg, add at least one bg
            num_bg = max(int(num_fg * bg_fg_ratio), 1)
            perm = torch.randperm(bg_loss.shape[0], device=bg_loss.device)[:num_bg]
            bg_loss = bg_loss[perm]

            return torch.cat([fg_loss, bg_loss], dim=0).mean()
        else:
            attri_loss = F.binary_cross_entropy_with_logits(logits, labels)
            attri_loss = attri_loss * self.num_attri_cat / 20.0
            return attri_loss


def make_roi_relation_loss_evaluator_deng(cfg, if_parent_model):

    loss_evaluator = RelationLoss_deng(
        cfg.MODEL.ATTRIBUTE_ON,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
        cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS,
        cfg.MODEL.ROI_RELATION_HEAD.REL_PROP,
        if_parent_model,
        cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX,
    )

    return loss_evaluator





class RelationLossComputation(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
        self,
        attri_on,
        num_attri_cat,
        max_num_attri,
        attribute_sampling,
        attribute_bgfg_ratio,
        use_label_smoothing,
        predicate_proportion,

    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        self.attri_on = attri_on
        self.num_attri_cat = num_attri_cat
        self.max_num_attri = max_num_attri
        self.attribute_sampling = attribute_sampling
        self.attribute_bgfg_ratio = attribute_bgfg_ratio
        self.use_label_smoothing = use_label_smoothing
        self.pred_weight = (1.0 / torch.FloatTensor([0.5,] + predicate_proportion)).cuda()

        if self.use_label_smoothing:
            self.criterion_loss = Label_Smoothing_Regression(e=0.01)
        else:
            self.criterion_loss = nn.CrossEntropyLoss()



    def __call__(self, proposals, rel_labels, relation_logits, refine_logits):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.

        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])

        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """
        if self.attri_on:
            if isinstance(refine_logits[0], (list, tuple)):
                refine_obj_logits, refine_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attri_on = False
                refine_obj_logits = refine_logits
        else:
            refine_obj_logits = refine_logits


        relation_logits = cat(relation_logits, dim=0)

        refine_obj_logits = cat(refine_obj_logits, dim=0)

        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        rel_labels = cat(rel_labels, dim=0)

        loss_relation = self.criterion_loss(relation_logits, rel_labels.long())
        loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())

        # The following code is used to calcaulate sampled attribute loss
        if self.attri_on:
            refine_att_logits = cat(refine_att_logits, dim=0)
            fg_attributes = cat([proposal.get_field("attributes") for proposal in proposals], dim=0)

            attribute_targets, fg_attri_idx = self.generate_attributes_target(fg_attributes)
            if float(fg_attri_idx.sum()) > 0:
                # have at least one bbox got fg attributes
                refine_att_logits = refine_att_logits[fg_attri_idx > 0]
                attribute_targets = attribute_targets[fg_attri_idx > 0]
            else:
                refine_att_logits = refine_att_logits[0].view(1, -1)
                attribute_targets = attribute_targets[0].view(1, -1)

            loss_refine_att = self.attribute_loss(refine_att_logits, attribute_targets,
                                             fg_bg_sample=self.attribute_sampling,
                                             bg_fg_ratio=self.attribute_bgfg_ratio)
            return loss_relation, (loss_refine_obj, loss_refine_att)
        else:
            return loss_relation, loss_refine_obj

    def generate_attributes_target(self, attributes):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        assert self.max_num_attri == attributes.shape[1]
        device = attributes.device
        num_obj = attributes.shape[0]

        fg_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch.zeros((num_obj, self.num_attri_cat), device=device).float()

        for idx in torch.nonzero(fg_attri_idx).squeeze(1).tolist():
            for k in range(self.max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, fg_attri_idx

    def attribute_loss(self, logits, labels, fg_bg_sample=True, bg_fg_ratio=3):
        if fg_bg_sample:
            loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').view(-1)
            fg_loss = loss_matrix[labels.view(-1) > 0]
            bg_loss = loss_matrix[labels.view(-1) <= 0]

            num_fg = fg_loss.shape[0]
            # if there is no fg, add at least one bg
            num_bg = max(int(num_fg * bg_fg_ratio), 1)
            perm = torch.randperm(bg_loss.shape[0], device=bg_loss.device)[:num_bg]
            bg_loss = bg_loss[perm]

            return torch.cat([fg_loss, bg_loss], dim=0).mean()
        else:
            attri_loss = F.binary_cross_entropy_with_logits(logits, labels)
            attri_loss = attri_loss * self.num_attri_cat / 20.0
            return attri_loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        target = target.view(-1)

        logpt = F.log_softmax(input)
        logpt = logpt.index_select(-1, target).diag()
        logpt = logpt.view(-1)
        pt = logpt.exp()

        logpt = logpt * self.alpha * (target > 0).float() + logpt * (1 - self.alpha) * (target <= 0).float()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()



def make_roi_relation_loss_evaluator(cfg):


    loss_evaluator = RelationLossComputation(
        cfg.MODEL.ATTRIBUTE_ON,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
        cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS,
        cfg.MODEL.ROI_RELATION_HEAD.REL_PROP,
    )


    return loss_evaluator