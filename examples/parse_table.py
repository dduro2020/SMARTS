import re

def parse_q_table_to_python(input_text, output_file="examples/q_table.py"):
    """
    Convierte una tabla Q en formato de texto a un archivo Python.

    Args:
        input_text (str): Tabla Q en formato de texto.
        output_file (str): Nombre del archivo Python de salida.
    """
    q_table = {}
    current_state = None

    # Procesar cada línea del texto de entrada
    for line in input_text.splitlines():
        # Detectar estados
        state_match = re.match(r"State: \(([^)]+)\)", line)
        if state_match:
            # Extraer el estado y convertirlo en una tupla
            state = tuple(map(float, state_match.group(1).split(", ")))
            q_table[state] = {}
            current_state = state
            continue

        # Detectar acciones y Q-valores
        action_match = re.match(r"\s+Action: ([\-0-9]+), Q-value: ([\d\.e\-]+)", line)
        if action_match and current_state is not None:
            # Extraer acción y Q-valor
            action = int(action_match.group(1))
            q_value = float(action_match.group(2))
            q_table[current_state][action] = q_value

    # Guardar como archivo Python
    with open(output_file, "w") as file:
        file.write("q_table = {\n")
        for state, actions in q_table.items():
            file.write(f"    {state}: {{\n")
            for action, value in actions.items():
                file.write(f"        {action}: {value},\n")
            file.write("    },\n")
        file.write("}\n")

    print(f"Q-table guardada en {output_file}")

# Ejemplo de uso
input_text = """
State: (1.75, 0.0)
  Action: -1, Q-value: 97.9267298747039
  Action: 0, Q-value: 5.490373914851162
  Action: 1, Q-value: 5.9390706293905975
State: (1.75, -0.1)
  Action: -1, Q-value: 111.0497498457664
  Action: 0, Q-value: 13.783641048812589
  Action: 1, Q-value: 6.783307828874712
State: (1.5, -0.2)
  Action: -1, Q-value: 93.17963054330865
  Action: 0, Q-value: 163.63875147448638
  Action: 1, Q-value: 86.00931348653276
State: (1.5, -0.1)
  Action: -1, Q-value: 148.1057600001609
  Action: 0, Q-value: 3.8882714338890367
  Action: 1, Q-value: 3.50790079155325
State: (1.5, 0.0)
  Action: -1, Q-value: 3.3290107635669655
  Action: 0, Q-value: 2.50796367441166
  Action: 1, Q-value: 5.6794859726804425
State: (1.5, 0.1)
  Action: -1, Q-value: 15.741888291261322
  Action: 0, Q-value: 5.836656391619408
  Action: 1, Q-value: 4.826951003631672
State: (1.5, 0.2)
  Action: -1, Q-value: 5.382870806675987
  Action: 0, Q-value: 2.0777597335248514
  Action: 1, Q-value: 1.803102113430418
State: (1.5, 0.30000000000000004)
  Action: -1, Q-value: 4.377543670910425
  Action: 0, Q-value: 1.2379113267339734
  Action: 1, Q-value: 0.9416584872495247
State: (1.5, 0.4)
  Action: -1, Q-value: 0.6261169732277261
  Action: 0, Q-value: 0.8387942543017818
  Action: 1, Q-value: 1.5243936956274382
State: (1.75, 0.4)
  Action: -1, Q-value: 0.9840734304200068
  Action: 0, Q-value: 0.32271567895893105
  Action: 1, Q-value: 0.2809353487489367
State: (2.0, 0.4)
  Action: -1, Q-value: 2.100842471853973
  Action: 0, Q-value: 0.9801850963240135
  Action: 1, Q-value: 0.5404547522928084
State: (2.0, 0.30000000000000004)
  Action: -1, Q-value: 3.1082724913397963
  Action: 0, Q-value: 1.8485976301987954
  Action: 1, Q-value: 1.498533217104775
State: (2.0, 0.5)
  Action: -1, Q-value: 0.6488784086080159
  Action: 0, Q-value: 1.2141900148101195
  Action: 1, Q-value: 0.42962660664307595
State: (2.0, 0.6000000000000001)
  Action: -1, Q-value: 0.6640482079771707
  Action: 0, Q-value: 0.1893457618507982
  Action: 1, Q-value: 0.0
State: (2.25, 0.6000000000000001)
  Action: -1, Q-value: 0.1765235966661052
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.08590564019579783
State: (2.5, 0.5)
  Action: -1, Q-value: 0.30884109466960796
  Action: 0, Q-value: 0.4690065401379431
  Action: 1, Q-value: 0.583627439772066
State: (2.5, 0.6000000000000001)
  Action: -1, Q-value: 0.3758695149636187
  Action: 0, Q-value: 0.3849528718840419
  Action: 1, Q-value: 0.09789522681835885
State: (2.75, 0.6000000000000001)
  Action: -1, Q-value: 0.5118652010157944
  Action: 0, Q-value: 0.18118130577473035
  Action: 1, Q-value: 0.2583530752804829
State: (2.75, 0.7000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.21485547556684417
  Action: 1, Q-value: 0.08994546541690339
State: (3.0, 0.7000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.09799563458422804
  Action: 1, Q-value: 0.3023271080495438
State: (3.0, 0.8)
  Action: -1, Q-value: 0.16030053002371175
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.1624198243707361
State: (3.25, 0.7000000000000001)
  Action: -1, Q-value: 0.1573644326707113
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.2834635416200961
State: (3.5, 0.8)
  Action: -1, Q-value: 0.11097517477795159
  Action: 0, Q-value: 0.05420613767008943
  Action: 1, Q-value: 0.2717793966677701
State: (3.5, 0.7000000000000001)
  Action: -1, Q-value: 0.1813919162544907
  Action: 0, Q-value: 0.12862149124168998
  Action: 1, Q-value: 0.1054592798681219
State: (3.75, 0.6000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.20757923901497943
State: (3.75, 0.7000000000000001)
  Action: -1, Q-value: 0.1144759046907219
  Action: 0, Q-value: 0.18750727651457733
  Action: 1, Q-value: 0.0
State: (4.0, 0.6000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.1816056105014825
  Action: 1, Q-value: 0.06087177319970231
State: (4.25, 0.6000000000000001)
  Action: -1, Q-value: 0.2638376160808601
  Action: 0, Q-value: 0.17579256145946123
  Action: 1, Q-value: 0.04519586280258704
State: (4.25, 0.5)
  Action: -1, Q-value: 0.10063264882796068
  Action: 0, Q-value: 0.1733879948249422
  Action: 1, Q-value: 0.25248523838650305
State: (4.25, 0.4)
  Action: -1, Q-value: 0.09065303268138394
  Action: 0, Q-value: 0.05620127090985199
  Action: 1, Q-value: 0.34217859021214586
State: (4.5, 0.5)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-1.75, 0.0)
  Action: -1, Q-value: 5.676156521641801
  Action: 0, Q-value: 5.262564130097117
  Action: 1, Q-value: 87.39271717585459
State: (-1.75, 0.1)
  Action: -1, Q-value: 15.68966480242074
  Action: 0, Q-value: 99.37291107422166
  Action: 1, Q-value: 4.965276590904902
State: (-1.75, -0.1)
  Action: -1, Q-value: 2.7005373076924224
  Action: 0, Q-value: 5.090119813429654
  Action: 1, Q-value: 5.723800729583321
State: (-1.75, 0.2)
  Action: -1, Q-value: 6.835875413533278
  Action: 0, Q-value: 0.5426445817537575
  Action: 1, Q-value: 0.8957579965751705
State: (-1.5, 0.2)
  Action: -1, Q-value: 45.55040238745184
  Action: 0, Q-value: 11.265056933232575
  Action: 1, Q-value: 149.40679329610558
State: (-1.5, 0.1)
  Action: -1, Q-value: 17.714846465235706
  Action: 0, Q-value: 4.472507753479464
  Action: 1, Q-value: 135.0225724949979
State: (-1.5, 0.0)
  Action: -1, Q-value: 0.6852085881435233
  Action: 0, Q-value: 11.34369410723972
  Action: 1, Q-value: 45.10378955306326
State: (-1.5, -0.1)
  Action: -1, Q-value: 1.5213490395783813
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.7814376550243606
State: (-1.5, -0.2)
  Action: -1, Q-value: 0.7587109582946628
  Action: 0, Q-value: 1.1083139204830343
  Action: 1, Q-value: 0.9392027411682975
State: (-1.5, -0.30000000000000004)
  Action: -1, Q-value: 1.0913365860839006
  Action: 0, Q-value: 0.8800731819299201
  Action: 1, Q-value: 0.48917031793241367
State: (-1.75, -0.2)
  Action: -1, Q-value: 1.1015452827245102
  Action: 0, Q-value: 3.0680496536706086
  Action: 1, Q-value: 0.1878246986874471
State: (-1.75, -0.30000000000000004)
  Action: -1, Q-value: 0.5272886538644734
  Action: 0, Q-value: 0.6033694281507473
  Action: 1, Q-value: 0.7821288460738598
State: (-1.75, -0.4)
  Action: -1, Q-value: 0.14322309955723983
  Action: 0, Q-value: 0.4023962206996169
  Action: 1, Q-value: 0.917289818129059
State: (-2.0, -0.4)
  Action: -1, Q-value: 0.629189051980801
  Action: 0, Q-value: 0.1069254019648883
  Action: 1, Q-value: 0.8054397229314578
State: (-2.0, -0.5)
  Action: -1, Q-value: 0.30376968021647427
  Action: 0, Q-value: 0.5848530504658629
  Action: 1, Q-value: 0.5162095535688874
State: (-2.0, -0.6000000000000001)
  Action: -1, Q-value: 0.27778781145244524
  Action: 0, Q-value: 0.11036961122832503
  Action: 1, Q-value: 0.16350860593535713
State: (-2.0, -0.7000000000000001)
  Action: -1, Q-value: 0.16383334108672132
  Action: 0, Q-value: 0.27483541418882723
  Action: 1, Q-value: 0.2158858715380305
State: (-2.25, -0.8)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.08119516491863527
State: (-2.5, -0.7000000000000001)
  Action: -1, Q-value: 0.15051772628401655
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.3016960760333059
State: (-2.5, -0.8)
  Action: -1, Q-value: 0.15319035026535482
  Action: 0, Q-value: 0.2246003562492664
  Action: 1, Q-value: 0.0
State: (-2.75, -0.8)
  Action: -1, Q-value: 0.13380680417347318
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-3.0, -0.9)
  Action: -1, Q-value: 0.06496414634086282
  Action: 0, Q-value: 0.1330399708900057
  Action: 1, Q-value: 0.06188401171777298
State: (-3.0, -1.0)
  Action: -1, Q-value: 0.13279543125645937
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-3.25, -1.1)
  Action: -1, Q-value: 0.23021452457801617
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-3.5, -1.2000000000000002)
  Action: -1, Q-value: 0.14192686887948844
  Action: 0, Q-value: 0.05448680129192375
  Action: 1, Q-value: 0.05392836256773603
State: (-3.75, -1.1)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.051056757558816813
  Action: 1, Q-value: 0.0
State: (-4.0, -1.1)
  Action: -1, Q-value: 0.04848583749994882
  Action: 0, Q-value: 0.04783094513663878
  Action: 1, Q-value: 0.15101384235614734
State: (-4.25, -1.2000000000000002)
  Action: -1, Q-value: 0.04597337760924018
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-4.25, -1.3)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: -3.243378921662054
  Action: 1, Q-value: 0.0
State: (-4.75, -1.3)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (1.5, -0.30000000000000004)
  Action: -1, Q-value: 3.520229634864156
  Action: 0, Q-value: 5.891235918495432
  Action: 1, Q-value: 169.3840244531321
State: (1.25, -0.2)
  Action: -1, Q-value: 41.397998471717386
  Action: 0, Q-value: 304.8929375101853
  Action: 1, Q-value: 26.10483930335116
State: (1.25, -0.1)
  Action: -1, Q-value: 165.81786840206223
  Action: 0, Q-value: 0.4454470647847899
  Action: 1, Q-value: 0.5437883632539064
State: (1.0, -0.2)
  Action: -1, Q-value: 388.18194004522445
  Action: 0, Q-value: 30.404883292684815
  Action: 1, Q-value: 3.7136959690751086
State: (1.0, -0.1)
  Action: -1, Q-value: 37.77219752478644
  Action: 0, Q-value: 2.0923781676895623
  Action: 1, Q-value: 1.3642766329302844
State: (1.0, 0.0)
  Action: -1, Q-value: 3.017533376215702
  Action: 0, Q-value: 1.8730740205711367
  Action: 1, Q-value: 1.4431000539896317
State: (1.0, -0.30000000000000004)
  Action: -1, Q-value: 425.5863200411104
  Action: 0, Q-value: 129.75271380521073
  Action: 1, Q-value: 75.95537765945589
State: (1.0, -0.4)
  Action: -1, Q-value: 76.96776114784805
  Action: 0, Q-value: 463.67708866524254
  Action: 1, Q-value: 329.1077864994256
State: (0.75, -0.4)
  Action: -1, Q-value: 229.26691166310925
  Action: 0, Q-value: 117.99757065363859
  Action: 1, Q-value: 654.2822102110109
State: (0.75, -0.5)
  Action: -1, Q-value: 0.28574615834508493
  Action: 0, Q-value: 4.174693638354497
  Action: 1, Q-value: 496.0038716013041
State: (0.5, -0.4)
  Action: -1, Q-value: 92.29388189857654
  Action: 0, Q-value: 60.20968444760075
  Action: 1, Q-value: 667.9858057383134
State: (0.5, -0.30000000000000004)
  Action: -1, Q-value: 245.69517827249882
  Action: 0, Q-value: 21.487146538983414
  Action: 1, Q-value: 748.5020554769644
State: (0.25, -0.4)
  Action: -1, Q-value: 5.4684962197055125
  Action: 0, Q-value: 10.623473485995055
  Action: 1, Q-value: 382.7927207602135
State: (0.25, -0.5)
  Action: -1, Q-value: 1.2856318642138507
  Action: 0, Q-value: 2.1828306579177292
  Action: 1, Q-value: 85.61960065387991
State: (0.0, -0.4)
  Action: -1, Q-value: 21.32600633040431
  Action: 0, Q-value: 67.6049940750718
  Action: 1, Q-value: 684.7361806919473
State: (0.0, -0.5)
  Action: -1, Q-value: 4.028403496235613
  Action: 0, Q-value: 7.908735742754344
  Action: 1, Q-value: 47.53720735669965
State: (-0.25, -0.5)
  Action: -1, Q-value: 1.1292970969492186
  Action: 0, Q-value: 1.5968460420725181
  Action: 1, Q-value: 0.0
State: (-0.5, -0.5)
  Action: -1, Q-value: 0.9162365868111496
  Action: 0, Q-value: 1.0168134507958988
  Action: 1, Q-value: 3.6681042926633705
State: (-0.5, -0.4)
  Action: -1, Q-value: 2.8691182369605333
  Action: 0, Q-value: 3.6258243898249565
  Action: 1, Q-value: 18.88760754907291
State: (-0.5, -0.30000000000000004)
  Action: -1, Q-value: 5.940391055094033
  Action: 0, Q-value: 8.446829404472942
  Action: 1, Q-value: 36.03794568429248
State: (-0.5, -0.2)
  Action: -1, Q-value: 8.315409561849808
  Action: 0, Q-value: 6.432444072896358
  Action: 1, Q-value: 70.01242797322391
State: (-0.5, -0.1)
  Action: -1, Q-value: 11.564526294895092
  Action: 0, Q-value: 22.31148286761983
  Action: 1, Q-value: 126.35621061576589
State: (-0.5, 0.0)
  Action: -1, Q-value: 13.72442212316875
  Action: 0, Q-value: 23.686857197613975
  Action: 1, Q-value: 222.8561388856968
State: (-0.75, -0.1)
  Action: -1, Q-value: 5.352633429713481
  Action: 0, Q-value: 3.055855396187681
  Action: 1, Q-value: 3.7475847108447837
State: (-0.75, -0.2)
  Action: -1, Q-value: 2.4067124673173126
  Action: 0, Q-value: 2.2836004876325577
  Action: 1, Q-value: 5.228258200518429
State: (-1.0, -0.30000000000000004)
  Action: -1, Q-value: 2.326123626614232
  Action: 0, Q-value: 2.2093101532493393
  Action: 1, Q-value: 4.448138657220379
State: (-1.0, -0.2)
  Action: -1, Q-value: 2.676812231340769
  Action: 0, Q-value: 3.72368207920544
  Action: 1, Q-value: 5.6155200907281975
State: (-1.0, -0.4)
  Action: -1, Q-value: 1.1547382419537242
  Action: 0, Q-value: 0.22241707285299606
  Action: 1, Q-value: 2.6369656913706545
State: (-1.25, -0.4)
  Action: -1, Q-value: 0.37920621929598425
  Action: 0, Q-value: 0.8449947992397275
  Action: 1, Q-value: 0.439202996620723
State: (-1.5, -0.4)
  Action: -1, Q-value: 0.5184615735395282
  Action: 0, Q-value: 1.7617678373593402
  Action: 1, Q-value: 0.7967776229997706
State: (-2.0, -0.30000000000000004)
  Action: -1, Q-value: 0.7222256660100014
  Action: 0, Q-value: 1.935098399251937
  Action: 1, Q-value: 1.4033378296490466
State: (-2.0, -0.2)
  Action: -1, Q-value: 1.673486066098309
  Action: 0, Q-value: 3.124414540447297
  Action: 1, Q-value: 2.7397215092700167
State: (-2.25, -0.4)
  Action: -1, Q-value: 0.4599695687426283
  Action: 0, Q-value: 1.5883130845652267
  Action: 1, Q-value: 0.3302732270260518
State: (-2.25, -0.5)
  Action: -1, Q-value: 0.17364514085230526
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.3973030052835418
State: (-2.5, -0.6000000000000001)
  Action: -1, Q-value: 0.34273214477616604
  Action: 0, Q-value: 0.11290227785143739
  Action: 1, Q-value: 0.1171214481837764
State: (-2.75, -0.9)
  Action: -1, Q-value: 0.07849615474758315
  Action: 0, Q-value: 0.0799483710757041
  Action: 1, Q-value: 0.13436488511878003
State: (-3.0, -0.8)
  Action: -1, Q-value: 0.23825107550090102
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.14809075729765542
State: (-3.0, -0.7000000000000001)
  Action: -1, Q-value: 0.5858846423265269
  Action: 0, Q-value: 0.22008986615318665
  Action: 1, Q-value: 0.0745726915450962
State: (-3.25, -0.7000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.41147983107573416
State: (-3.5, -0.6000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.5499266857702722
  Action: 1, Q-value: 0.28213464328236537
State: (-3.5, -0.5)
  Action: -1, Q-value: 0.10809870617204315
  Action: 0, Q-value: 0.20478314901204814
  Action: 1, Q-value: 0.43924545168743595
State: (-3.75, -0.6000000000000001)
  Action: -1, Q-value: 0.4500946590859246
  Action: 0, Q-value: 0.24451314415074166
  Action: 1, Q-value: 0.052582632258555584
State: (-3.75, -0.5)
  Action: -1, Q-value: 0.05130464643030552
  Action: 0, Q-value: 0.08474654783877675
  Action: 1, Q-value: 0.25691984353344033
State: (-4.0, -0.6000000000000001)
  Action: -1, Q-value: 0.09757714607322393
  Action: 0, Q-value: 0.47419246725929687
  Action: 1, Q-value: 0.049159821019333194
State: (-4.25, -0.7000000000000001)
  Action: -1, Q-value: 0.04698615465514561
  Action: 0, Q-value: 0.2897494391136991
  Action: 1, Q-value: 0.044673948778003125
State: (-4.25, -0.8)
  Action: -1, Q-value: 0.0461722125860025
  Action: 0, Q-value: 0.19069942059765868
  Action: 1, Q-value: 0.05530803674844381
State: (-4.5, -0.8)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (3.75, 0.0)
  Action: -1, Q-value: 2.7237332052485765
  Action: 0, Q-value: 1.7958362257643412
  Action: 1, Q-value: 2.532420512357428
State: (3.75, 0.1)
  Action: -1, Q-value: 2.509157555388061
  Action: 0, Q-value: 2.417853621718129
  Action: 1, Q-value: 2.3402124289033805
State: (3.75, -0.1)
  Action: -1, Q-value: 2.1790083607715287
  Action: 0, Q-value: 2.733049624139834
  Action: 1, Q-value: 2.421309215990121
State: (3.75, 0.2)
  Action: -1, Q-value: 2.4607954950450925
  Action: 0, Q-value: 1.1576145511070306
  Action: 1, Q-value: 0.541923210117476
State: (3.75, -0.2)
  Action: -1, Q-value: 0.4290964339946696
  Action: 0, Q-value: 0.36541586820786953
  Action: 1, Q-value: 2.441056575000199
State: (3.75, 0.30000000000000004)
  Action: -1, Q-value: 0.9591936857725433
  Action: 0, Q-value: 0.6621523528589834
  Action: 1, Q-value: 0.4300699043703914
State: (4.0, 0.30000000000000004)
  Action: -1, Q-value: 1.181617958732295
  Action: 0, Q-value: 0.23078810393910057
  Action: 1, Q-value: 0.2737413793762081
State: (4.0, 0.2)
  Action: -1, Q-value: 1.8852889080823279
  Action: 0, Q-value: 0.8959032883028292
  Action: 1, Q-value: 0.5909769545230248
State: (4.0, 0.1)
  Action: -1, Q-value: 1.24877927688119
  Action: 0, Q-value: 0.46541405492141824
  Action: 1, Q-value: 1.8593484926704305
State: (4.25, 0.30000000000000004)
  Action: -1, Q-value: 0.1879876723036636
  Action: 0, Q-value: 0.21359879214486288
  Action: 1, Q-value: 0.3935083488299576
State: (1.75, -0.2)
  Action: -1, Q-value: 30.1049717498082
  Action: 0, Q-value: 14.026270420615633
  Action: 1, Q-value: 104.46612246215558
State: (1.5, -0.4)
  Action: -1, Q-value: 0.8770523121013478
  Action: 0, Q-value: 1.7949426017579282
  Action: 1, Q-value: 7.34893861145964
State: (1.25, -0.4)
  Action: -1, Q-value: 0.3811016542621567
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 16.662663319391
State: (1.0, -0.5)
  Action: -1, Q-value: 1.3065745329069833
  Action: 0, Q-value: 233.51432750834664
  Action: 1, Q-value: 14.636830761444788
State: (0.5, -0.5)
  Action: -1, Q-value: 4.6565541281862295
  Action: 0, Q-value: 22.73164025085751
  Action: 1, Q-value: 402.4961495218849
State: (0.0, -0.30000000000000004)
  Action: -1, Q-value: 331.21005098531504
  Action: 0, Q-value: 748.513642781404
  Action: 1, Q-value: 1585.3895807112513
State: (0.0, -0.2)
  Action: -1, Q-value: 1191.4573631700007
  Action: 0, Q-value: 1498.9909698638494
  Action: 1, Q-value: 1796.5238445654138
State: (0.0, -0.1)
  Action: -1, Q-value: 1608.5359634930128
  Action: 0, Q-value: 1686.0589964279773
  Action: 1, Q-value: 1989.1042300816116
State: (0.0, 0.0)
  Action: -1, Q-value: 1801.1358252628447
  Action: 0, Q-value: 1999.9999191628526
  Action: 1, Q-value: 1803.6983680182743
State: (-0.25, -0.1)
  Action: -1, Q-value: 14.675269127441654
  Action: 0, Q-value: 7.273919962465549
  Action: 1, Q-value: 5.710432394226824
State: (-0.25, -0.2)
  Action: -1, Q-value: 12.129860602092403
  Action: 0, Q-value: 5.60609094311602
  Action: 1, Q-value: 8.314931181213586
State: (-0.25, -0.30000000000000004)
  Action: -1, Q-value: 12.507645916401426
  Action: 0, Q-value: 6.205024657957667
  Action: 1, Q-value: 3.631069750341244
State: (-0.75, -0.4)
  Action: -1, Q-value: 0.2923019627720848
  Action: 0, Q-value: 0.25718270312894276
  Action: 1, Q-value: 1.9066112386124616
State: (-1.0, -0.1)
  Action: -1, Q-value: 4.872947106381114
  Action: 0, Q-value: 6.081694169140386
  Action: 1, Q-value: 5.257530075847958
State: (-1.0, 0.0)
  Action: -1, Q-value: 2.8908758485082355
  Action: 0, Q-value: 2.5983961344869897
  Action: 1, Q-value: 6.351585115950465
State: (-1.25, -0.30000000000000004)
  Action: -1, Q-value: 0.9597578994711712
  Action: 0, Q-value: 0.22290156073476433
  Action: 1, Q-value: 0.5326614882935697
State: (-2.5, -0.5)
  Action: -1, Q-value: 0.39746271056743526
  Action: 0, Q-value: 1.4657569489373758
  Action: 1, Q-value: 0.6380607993168644
State: (-2.5, -0.4)
  Action: -1, Q-value: 1.5580933032929127
  Action: 0, Q-value: 0.534401921344536
  Action: 1, Q-value: 0.3908773222609573
State: (-2.75, -0.5)
  Action: -1, Q-value: 0.3760921469623763
  Action: 0, Q-value: 1.0008571953412955
  Action: 1, Q-value: 0.3677879104809635
State: (-2.75, -0.6000000000000001)
  Action: -1, Q-value: 0.35681924997094006
  Action: 0, Q-value: 0.06928324190115892
  Action: 1, Q-value: 0.11966682518521057
State: (-3.25, -0.9)
  Action: -1, Q-value: 0.1538865506467955
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.12631790235030332
State: (-3.5, -0.8)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.2071775539629819
State: (-3.5, -0.7000000000000001)
  Action: -1, Q-value: 0.10782382372536262
  Action: 0, Q-value: 0.6477235676698926
  Action: 1, Q-value: 0.0856131952311453
State: (-3.75, -0.7000000000000001)
  Action: -1, Q-value: 0.10185054203683029
  Action: 0, Q-value: 0.07116753805792357
  Action: 1, Q-value: 0.5174611815417669
State: (-4.25, -0.6000000000000001)
  Action: -1, Q-value: 0.3363314440340674
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (0.0, 0.1)
  Action: -1, Q-value: 1992.0632995643525
  Action: 0, Q-value: 1731.321918644324
  Action: 1, Q-value: 1570.405291755175
State: (0.0, 0.2)
  Action: -1, Q-value: 1803.3016118895468
  Action: 0, Q-value: 1361.4034644364042
  Action: 1, Q-value: 1150.27355039491
State: (0.0, 0.30000000000000004)
  Action: -1, Q-value: 1582.6534830122976
  Action: 0, Q-value: 698.2792792512303
  Action: 1, Q-value: 156.3960348464508
State: (0.0, 0.4)
  Action: -1, Q-value: 569.0377506252886
  Action: 0, Q-value: 11.639271314185134
  Action: 1, Q-value: 4.041830155600619
State: (0.0, 0.5)
  Action: -1, Q-value: 39.014132166789395
  Action: 0, Q-value: 2.4422762428546885
  Action: 1, Q-value: 2.4570902380816015
State: (0.25, 0.4)
  Action: -1, Q-value: 0.742886576377709
  Action: 0, Q-value: 2.465094787492394
  Action: 1, Q-value: 0.7346564148389563
State: (0.5, 0.4)
  Action: -1, Q-value: 1.1573073716476245
  Action: 0, Q-value: 1.914405683991433
  Action: 1, Q-value: 2.5420860818500244
State: (0.5, 0.30000000000000004)
  Action: -1, Q-value: 7.848563529740202
  Action: 0, Q-value: 3.036094330442936
  Action: 1, Q-value: 2.1657593396712667
State: (0.75, 0.4)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.504082080782178
State: (0.75, 0.5)
  Action: -1, Q-value: 1.182031796373062
  Action: 0, Q-value: 0.2286223812876303
  Action: 1, Q-value: 0.0
State: (1.0, 0.5)
  Action: -1, Q-value: 1.62115390788406
  Action: 0, Q-value: 0.8160100010622254
  Action: 1, Q-value: 1.0863440887344828
State: (1.0, 0.4)
  Action: -1, Q-value: 2.979831311676518
  Action: 0, Q-value: 0.8386834505601475
  Action: 1, Q-value: 1.746808944703633
State: (1.25, 0.4)
  Action: -1, Q-value: 2.844813493473858
  Action: 0, Q-value: 0.45326184146885695
  Action: 1, Q-value: 0.4760915634812576
State: (1.5, 0.5)
  Action: -1, Q-value: 0.8303713405187049
  Action: 0, Q-value: 0.4659566527793154
  Action: 1, Q-value: 1.5637784956020326
State: (1.75, 0.6000000000000001)
  Action: -1, Q-value: 0.9852964173513267
  Action: 0, Q-value: 0.23846690275741173
  Action: 1, Q-value: 0.10728842338125683
State: (3.25, 0.9)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.11609550602230734
State: (3.5, 1.0)
  Action: -1, Q-value: 0.05564033274282401
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.05593658347439279
State: (3.5, 1.1)
  Action: -1, Q-value: 0.052846234094955626
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (3.75, 1.0)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.06795087009011586
  Action: 1, Q-value: 0.050327798312060926
State: (4.0, 1.1)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.04783098201048136
State: (4.25, 1.2000000000000002)
  Action: -1, Q-value: 0.04538763347621757
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (4.5, 1.1)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-1.75, 0.30000000000000004)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.2345406260253265
  Action: 1, Q-value: 0.11835520378350912
State: (-1.5, 0.30000000000000004)
  Action: -1, Q-value: 0.8700923638022175
  Action: 0, Q-value: 167.3166745613556
  Action: 1, Q-value: 2.615566661184501
State: (-1.25, 0.4)
  Action: -1, Q-value: 90.9026988453391
  Action: 0, Q-value: 0.1659812167625417
  Action: 1, Q-value: 0.17150767751837437
State: (-1.0, 0.30000000000000004)
  Action: -1, Q-value: 308.7434980647528
  Action: 0, Q-value: 91.81100698794876
  Action: 1, Q-value: 1.5058310309433527
State: (-1.0, 0.2)
  Action: -1, Q-value: 104.79636784260552
  Action: 0, Q-value: 345.74879006281725
  Action: 1, Q-value: 169.80028265449099
State: (-1.0, 0.1)
  Action: -1, Q-value: 3.1465225913047576
  Action: 0, Q-value: 3.3986111776949977
  Action: 1, Q-value: 224.13615863400804
State: (-0.75, 0.30000000000000004)
  Action: -1, Q-value: 167.2101203518007
  Action: 0, Q-value: 16.446300345416432
  Action: 1, Q-value: 1.7792275245204943
State: (-0.5, 0.4)
  Action: -1, Q-value: 436.45301876848197
  Action: 0, Q-value: 3.812404674567727
  Action: 1, Q-value: 6.644434169434829
State: (-0.5, 0.30000000000000004)
  Action: -1, Q-value: 877.003052836778
  Action: 0, Q-value: 460.13353852140756
  Action: 1, Q-value: 156.6896846180978
State: (-0.25, 0.4)
  Action: -1, Q-value: 420.12372882937973
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 7.253467196357408
State: (0.25, 0.2)
  Action: -1, Q-value: 219.5007543849456
  Action: 0, Q-value: 1.2305561448653493
  Action: 1, Q-value: 6.169882975476252
State: (0.5, 0.5)
  Action: -1, Q-value: 2.7086324738396423
  Action: 0, Q-value: 0.5406837520577561
  Action: 1, Q-value: 1.2001445135815016
State: (0.75, 0.6000000000000001)
  Action: -1, Q-value: 0.29782482371502256
  Action: 0, Q-value: 0.9632060803073549
  Action: 1, Q-value: 0.561964064455291
State: (1.0, 0.30000000000000004)
  Action: -1, Q-value: 2.430024049889241
  Action: 0, Q-value: 5.164706203062937
  Action: 1, Q-value: 2.4092589544083722
State: (1.25, 0.30000000000000004)
  Action: -1, Q-value: 0.15361031436525913
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 2.0980131539544646
State: (1.25, 0.2)
  Action: -1, Q-value: 1.107381582633217
  Action: 0, Q-value: 0.7230466979759715
  Action: 1, Q-value: 0.7038922542705728
State: (1.75, 0.5)
  Action: -1, Q-value: 1.2254676575215755
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.37157038581515467
State: (2.0, 0.2)
  Action: -1, Q-value: 4.935213776601293
  Action: 0, Q-value: 2.681565111380673
  Action: 1, Q-value: 2.5890016072656112
State: (2.25, 0.4)
  Action: -1, Q-value: 1.1311080274242513
  Action: 0, Q-value: 0.5601279250451197
  Action: 1, Q-value: 0.42991451776045997
State: (2.5, 0.4)
  Action: -1, Q-value: 0.8469719998516014
  Action: 0, Q-value: 1.1868841906547896
  Action: 1, Q-value: 0.7903829427195794
State: (2.5, 0.30000000000000004)
  Action: -1, Q-value: 1.848857322737989
  Action: 0, Q-value: 0.6512621170143877
  Action: 1, Q-value: 1.0902252739425684
State: (2.75, 0.4)
  Action: -1, Q-value: 0.4708983903332485
  Action: 0, Q-value: 0.28062439004089546
  Action: 1, Q-value: 0.6468417905688681
State: (2.75, 0.30000000000000004)
  Action: -1, Q-value: 0.09412549492294785
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.7033419587062035
State: (3.0, 0.5)
  Action: -1, Q-value: 0.6757256069130246
  Action: 0, Q-value: 0.359028274272545
  Action: 1, Q-value: 0.49089795736461356
State: (3.0, 0.4)
  Action: -1, Q-value: 0.7100574846856502
  Action: 0, Q-value: 0.14876383842628638
  Action: 1, Q-value: 0.5810188765147537
State: (3.25, 0.5)
  Action: -1, Q-value: 0.13117433061806325
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.2415049540724317
State: (3.25, 0.6000000000000001)
  Action: -1, Q-value: 0.21650459372243003
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.271737371021486
State: (3.5, 0.5)
  Action: -1, Q-value: 0.15146311476431645
  Action: 0, Q-value: 0.3439499786339599
  Action: 1, Q-value: 0.10974080381094625
State: (3.5, 0.6000000000000001)
  Action: -1, Q-value: 0.23377522050514066
  Action: 0, Q-value: 0.0727037534378932
  Action: 1, Q-value: 0.0
State: (4.0, 0.7000000000000001)
  Action: -1, Q-value: 0.17853013131889278
  Action: 0, Q-value: 0.11012037391804619
  Action: 1, Q-value: 0.0
State: (3.5, -0.2)
  Action: -1, Q-value: 2.2224069850360815
  Action: 0, Q-value: 3.454071360469197
  Action: 1, Q-value: 2.464873642704559
State: (3.5, -0.30000000000000004)
  Action: -1, Q-value: 0.5313192049654313
  Action: 0, Q-value: 0.9659964957755833
  Action: 1, Q-value: 2.6496002211172653
State: (3.5, -0.4)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.18647834265340946
  Action: 1, Q-value: 0.5380717594616448
State: (3.25, -0.30000000000000004)
  Action: -1, Q-value: 1.217286698138945
  Action: 0, Q-value: 4.313371203050018
  Action: 1, Q-value: 2.1821830220101863
State: (3.25, -0.4)
  Action: -1, Q-value: 0.29202008832075427
  Action: 0, Q-value: 0.5349924709608764
  Action: 1, Q-value: 2.1817473526206945
State: (3.0, -0.30000000000000004)
  Action: -1, Q-value: 0.9466726726784249
  Action: 0, Q-value: 2.1726246534310523
  Action: 1, Q-value: 4.972556542687929
State: (3.0, -0.4)
  Action: -1, Q-value: 0.41135579629135216
  Action: 0, Q-value: 0.3487840992713742
  Action: 1, Q-value: 1.2934168366916974
State: (3.0, -0.5)
  Action: -1, Q-value: 0.37861024992032255
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.14508765388565525
State: (3.0, -0.6000000000000001)
  Action: -1, Q-value: 0.08252531544096392
  Action: 0, Q-value: 0.07216669227051563
  Action: 1, Q-value: 0.3471999892431719
State: (2.75, -0.6000000000000001)
  Action: -1, Q-value: 0.19831761721566338
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (2.5, -0.7000000000000001)
  Action: -1, Q-value: 0.20555780231858767
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0793323431682575
State: (2.5, -0.6000000000000001)
  Action: -1, Q-value: 0.09742848387765234
  Action: 0, Q-value: 0.3096582809557738
  Action: 1, Q-value: 0.8818712225959238
State: (2.25, -0.8)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.09453439180688913
  Action: 1, Q-value: 0.246687844009518
State: (2.0, -0.8)
  Action: -1, Q-value: 0.11193928864266729
  Action: 0, Q-value: 0.2980514906010403
  Action: 1, Q-value: 0.17580398666781216
State: (1.75, -0.8)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.12113354910635699
State: (1.5, -0.7000000000000001)
  Action: -1, Q-value: 0.4776090573248979
  Action: 0, Q-value: 0.16335012157295925
  Action: 1, Q-value: 0.19509800060063745
State: (1.5, -0.8)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.16598180955661426
  Action: 1, Q-value: 0.45193284205391016
State: (1.25, -0.7000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.18374529453433297
  Action: 1, Q-value: 0.0
State: (1.0, -0.7000000000000001)
  Action: -1, Q-value: 0.40727765890217643
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (1.0, -0.8)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.7137486638943169
  Action: 1, Q-value: 0.0
State: (0.75, -0.8)
  Action: -1, Q-value: 0.8685394071006467
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (0.5, -0.9)
  Action: -1, Q-value: 1.9639752846375207
  Action: 0, Q-value: 1.954185317222113
  Action: 1, Q-value: 0.42857274364180853
State: (0.5, -0.8)
  Action: -1, Q-value: 1.3952588014638156
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (0.0, -0.9)
  Action: -1, Q-value: 3.039880518759152
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (0.0, -1.0)
  Action: -1, Q-value: 1.2570698644159273
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-0.25, -1.1)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.4472013940321502
State: (-0.5, -1.0)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.2748642047177313
  Action: 1, Q-value: 0.3117115932832304
State: (-0.5, -0.9)
  Action: -1, Q-value: 0.5637728979881937
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-1.0, -1.0)
  Action: -1, Q-value: 0.4533707010240342
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-1.0, -1.1)
  Action: -1, Q-value: 0.3077964620459258
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.16334980414443726
State: (-1.25, -1.0)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.1410031661761064
  Action: 1, Q-value: 0.0
State: (-1.5, -1.0)
  Action: -1, Q-value: 0.11073889857200739
  Action: 0, Q-value: 0.12404557625139062
  Action: 1, Q-value: 0.0
State: (-1.75, -1.1)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.09906144461613593
State: (-2.0, -1.0)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.17403571789696312
State: (-2.25, -0.9)
  Action: -1, Q-value: 0.08183586849341849
  Action: 0, Q-value: 0.17423247359299865
  Action: 1, Q-value: 0.0
State: (-2.5, -0.9)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.27103087606632775
  Action: 1, Q-value: 0.0
State: (-3.25, -0.8)
  Action: -1, Q-value: 0.13428078469755947
  Action: 0, Q-value: 0.08434802989598365
  Action: 1, Q-value: 0.5712563866500806
State: (-4.0, -0.7000000000000001)
  Action: -1, Q-value: 0.2936912063778744
  Action: 0, Q-value: 0.04938886341635183
  Action: 1, Q-value: 0.0
State: (3.5, 0.0)
  Action: -1, Q-value: 2.4240374629303596
  Action: 0, Q-value: 0.3450810058918364
  Action: 1, Q-value: 0.530961968521757
State: (3.5, -0.1)
  Action: -1, Q-value: 1.105827693581027
  Action: 0, Q-value: 3.051212471467152
  Action: 1, Q-value: 1.5787847445148329
State: (3.5, 0.1)
  Action: -1, Q-value: 1.084366630758502
  Action: 0, Q-value: 0.11285697770586939
  Action: 1, Q-value: 0.6010091676210803
State: (3.25, -0.2)
  Action: -1, Q-value: 4.06053634093876
  Action: 0, Q-value: 0.10802212942407917
  Action: 1, Q-value: 1.4221237077899112
State: (3.0, -0.2)
  Action: -1, Q-value: 5.630919782368264
  Action: 0, Q-value: 2.5056691599353953
  Action: 1, Q-value: 0.9263653090831823
State: (3.0, -0.1)
  Action: -1, Q-value: 3.052864340816283
  Action: 0, Q-value: 0.36021301051244786
  Action: 1, Q-value: 0.7467877510310779
State: (2.75, -0.2)
  Action: -1, Q-value: 2.2271076100459917
  Action: 0, Q-value: 10.566269570898612
  Action: 1, Q-value: 2.55429393875687
State: (2.75, -0.1)
  Action: -1, Q-value: 2.8947222761946003
  Action: 0, Q-value: 1.0236059645140776
  Action: 1, Q-value: 0.6931620718532607
State: (2.75, 0.0)
  Action: -1, Q-value: 1.6526472283792735
  Action: 0, Q-value: 1.8167498255602266
  Action: 1, Q-value: 2.8591779033536193
State: (2.75, 0.1)
  Action: -1, Q-value: 2.6631248575835356
  Action: 0, Q-value: 1.9330378377084814
  Action: 1, Q-value: 0.8205641586812561
State: (2.75, 0.2)
  Action: -1, Q-value: 1.267366966583531
  Action: 0, Q-value: 0.32570455103400164
  Action: 1, Q-value: 0.27604253147447666
State: (3.0, 0.2)
  Action: -1, Q-value: 1.8787390061183316
  Action: 0, Q-value: 28.234556539300574
  Action: 1, Q-value: 0.40368632641689256
State: (3.0, 0.1)
  Action: -1, Q-value: 0.4206436321951707
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 12.696355000106113
State: (3.0, 0.0)
  Action: -1, Q-value: 1.9972178931199602
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.19472993520894455
State: (2.75, -0.30000000000000004)
  Action: -1, Q-value: 0.5948485175113593
  Action: 0, Q-value: 0.5921336440605216
  Action: 1, Q-value: 8.08767022902154
State: (2.5, -0.4)
  Action: -1, Q-value: 1.5868782881245185
  Action: 0, Q-value: 0.1574856891474623
  Action: 1, Q-value: 0.2666516732070473
State: (2.5, -0.30000000000000004)
  Action: -1, Q-value: 1.2411495834068533
  Action: 0, Q-value: 2.5965673097841786
  Action: 1, Q-value: 1.3765091233116353
State: (2.5, -0.5)
  Action: -1, Q-value: 0.5454393596239667
  Action: 0, Q-value: 2.472343673518294
  Action: 1, Q-value: 0.9698531445340338
State: (2.25, -0.5)
  Action: -1, Q-value: 0.5268193668047799
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 3.725514417687771
State: (2.25, -0.6000000000000001)
  Action: -1, Q-value: 0.21755969468854872
  Action: 0, Q-value: 0.3622369612016439
  Action: 1, Q-value: 0.43390055724234605
State: (2.0, -0.7000000000000001)
  Action: -1, Q-value: 0.24572130409372112
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.49874456588134614
State: (1.75, -0.9)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.15478857735710502
State: (0.25, -1.0)
  Action: -1, Q-value: 2.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 2.3600000000000003
State: (-0.5, -1.1)
  Action: -1, Q-value: 0.35469385242211887
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-0.5, -1.2000000000000002)
  Action: -1, Q-value: 0.25091276929731593
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.2580243762532878
State: (-0.75, -1.3)
  Action: -1, Q-value: 0.19054367418768858
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-1.0, -1.4000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.15135435290438604
  Action: 1, Q-value: 0.0
State: (-1.5, -1.4000000000000001)
  Action: -1, Q-value: 0.12555502461550608
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-1.5, -1.5)
  Action: -1, Q-value: 0.10618544941319433
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-2.0, -1.6)
  Action: -1, Q-value: 0.09120157775656043
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-2.25, -1.7000000000000002)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.07933222450887818
  Action: 1, Q-value: 0.0
State: (-2.5, -1.7000000000000002)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.10695050397960744
State: (1.75, -1.6)
  Action: -1, Q-value: 0.06338596297976061
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-3.25, -1.7000000000000002)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.05746795966360514
  Action: 1, Q-value: 0.0
State: (-3.5, -1.7000000000000002)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.05258262524783586
State: (-3.75, -1.6)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.048708345495780354
  Action: 1, Q-value: 0.0
State: (-4.0, -1.6)
  Action: -1, Q-value: 0.04558109031923506
  Action: 0, Q-value: 0.04538759934135203
  Action: 1, Q-value: 0.0
State: (-4.5, -1.6)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (1.75, -0.30000000000000004)
  Action: -1, Q-value: 0.13548256373721562
  Action: 0, Q-value: 0.23629848743578302
  Action: 1, Q-value: 35.49705173639856
State: (1.0, 0.1)
  Action: -1, Q-value: 1.7819254123199222
  Action: 0, Q-value: 1.8301933281472342
  Action: 1, Q-value: 4.656801071716614
State: (1.0, 0.2)
  Action: -1, Q-value: 3.404914003843605
  Action: 0, Q-value: 3.0839522742552035
  Action: 1, Q-value: 5.056160832313493
State: (2.25, 0.30000000000000004)
  Action: -1, Q-value: 1.394551552437793
  Action: 0, Q-value: 0.45905185938962356
  Action: 1, Q-value: 0.7760710803020785
State: (4.5, 0.4)
  Action: -1, Q-value: 0.1034174193144494
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (2.75, 0.5)
  Action: -1, Q-value: 0.3543799427223596
  Action: 0, Q-value: 0.6403703623389377
  Action: 1, Q-value: 0.33187542707872814
State: (3.0, 0.6000000000000001)
  Action: -1, Q-value: 0.1354221687970734
  Action: 0, Q-value: 0.5540313473744508
  Action: 1, Q-value: 0.2561254813900112
State: (3.25, 0.30000000000000004)
  Action: -1, Q-value: 0.4347941741107352
  Action: 0, Q-value: 0.7778125393647859
  Action: 1, Q-value: 0.4385276804674172
State: (3.25, 0.2)
  Action: -1, Q-value: 0.3515259954591431
  Action: 0, Q-value: 1.1508634906539865
  Action: 1, Q-value: 0.6756112390184315
State: (3.25, 0.1)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 498.0444696686782
State: (3.5, 0.30000000000000004)
  Action: -1, Q-value: 1.4869635498246079
  Action: 0, Q-value: 0.3816561375347541
  Action: 1, Q-value: 0.5182278227713493
State: (3.5, 0.4)
  Action: -1, Q-value: 0.6269882395334184
  Action: 0, Q-value: 0.3597455416513172
  Action: 1, Q-value: 0.29523810393096434
State: (3.5, 0.2)
  Action: -1, Q-value: 0.32928163890643786
  Action: 0, Q-value: 0.5534070625395522
  Action: 1, Q-value: 1.1382857224966982
State: (4.25, 0.1)
  Action: -1, Q-value: 0.7862687213643161
  Action: 0, Q-value: 2.212609826769254
  Action: 1, Q-value: 1.5408669901187693
State: (4.25, 0.0)
  Action: -1, Q-value: 0.3009478703473966
  Action: 0, Q-value: 0.20803702797048806
  Action: 1, Q-value: 1.4113388975634868
State: (4.25, -0.1)
  Action: -1, Q-value: 0.1470940827198592
  Action: 0, Q-value: 0.04848586472639976
  Action: 1, Q-value: 0.4061144904154027
State: (4.0, 0.0)
  Action: -1, Q-value: 0.14997979799858271
  Action: 0, Q-value: 0.10472824600453097
  Action: 1, Q-value: 1.8193750482410158
State: (4.25, 0.2)
  Action: -1, Q-value: 1.899006901269136
  Action: 0, Q-value: 1.0071364909949048
  Action: 1, Q-value: 0.4059522692059271
State: (4.5, 0.30000000000000004)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (0.25, 0.30000000000000004)
  Action: -1, Q-value: 7.802905762645169
  Action: 0, Q-value: 2.3610868981079167
  Action: 1, Q-value: 2.4998659234713325
State: (0.5, 0.2)
  Action: -1, Q-value: 11.111376698307224
  Action: 0, Q-value: 10.993299418146117
  Action: 1, Q-value: 5.425819464424626
State: (0.5, 0.1)
  Action: -1, Q-value: 7.369259419629779
  Action: 0, Q-value: 5.817296087035169
  Action: 1, Q-value: 12.084162507806937
State: (0.5, 0.0)
  Action: -1, Q-value: 5.78066572975977
  Action: 0, Q-value: 5.691378345556505
  Action: 1, Q-value: 12.353969552275872
State: (0.5, -0.1)
  Action: -1, Q-value: 749.6905018334257
  Action: 0, Q-value: 326.63307999576443
  Action: 1, Q-value: 12.444058646654758
State: (1.25, 0.5)
  Action: -1, Q-value: 0.7649204063744081
  Action: 0, Q-value: 0.6486729938991905
  Action: 1, Q-value: 0.5511084304768962
State: (1.5, 0.6000000000000001)
  Action: -1, Q-value: 1.6218998873088184
  Action: 0, Q-value: 0.1255552557484578
  Action: 1, Q-value: 0.18092438552283874
State: (2.25, 0.5)
  Action: -1, Q-value: 0.7053209122031252
  Action: 0, Q-value: 0.14349104283389227
  Action: 1, Q-value: 0.11979735966456613
State: (3.5, 0.9)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.22838555463173332
  Action: 1, Q-value: 0.06244065014793393
State: (3.75, 0.9)
  Action: -1, Q-value: 0.0515551040159279
  Action: 0, Q-value: 0.05105679877677652
  Action: 1, Q-value: 0.2196837644516778
State: (4.0, 1.0)
  Action: -1, Q-value: 0.24486036188516644
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (4.25, 0.9)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.1745089046415738
State: (4.25, 1.0)
  Action: -1, Q-value: -1.9642613066077166
  Action: 0, Q-value: -1.9285111916303346
  Action: 1, Q-value: 0.045195869162193356
State: (4.5, 0.9)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (1.75, 0.1)
  Action: -1, Q-value: 7.124576668865876
  Action: 0, Q-value: 5.372441470517022
  Action: 1, Q-value: 5.117074164600311
State: (1.75, 0.2)
  Action: -1, Q-value: 5.459009974914461
  Action: 0, Q-value: 2.0723845183343528
  Action: 1, Q-value: 0.8654095355038539
State: (4.25, 0.7000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.04698617605404881
  Action: 1, Q-value: 0.04558112488038072
State: (4.25, 0.8)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-2.5, -0.1)
  Action: -1, Q-value: 0.9970338273350234
  Action: 0, Q-value: 0.5334733132654774
  Action: 1, Q-value: 0.6984045089774953
State: (1.75, 0.30000000000000004)
  Action: -1, Q-value: 2.053679746597123
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.5937534922595882
State: (3.0, 0.30000000000000004)
  Action: -1, Q-value: 0.7568832960647031
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.21554592530184433
State: (3.75, 0.4)
  Action: -1, Q-value: 0.5514248127008297
  Action: 0, Q-value: 0.28011797536960326
  Action: 1, Q-value: 0.13250383275452682
State: (4.0, 0.4)
  Action: -1, Q-value: 0.4435570732915209
  Action: 0, Q-value: 0.23668761448893666
  Action: 1, Q-value: 0.20986001880440933
State: (4.0, 0.5)
  Action: -1, Q-value: 0.09293156941132906
  Action: 0, Q-value: 0.06681729091660797
  Action: 1, Q-value: 0.1890052650324272
State: (4.5, 0.6000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-1.5, 0.4)
  Action: -1, Q-value: 0.38553208501672326
  Action: 0, Q-value: 11.86737387129185
  Action: 1, Q-value: 0.40422612034052485
State: (-1.5, 0.5)
  Action: -1, Q-value: 0.3990102536901201
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.14295822509521897
State: (-1.5, 0.6000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.1559344124713605
  Action: 1, Q-value: 0.0
State: (-1.25, 0.6000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.17150767656069046
State: (-1.0, 0.7000000000000001)
  Action: -1, Q-value: 0.19413599123431713
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-1.0, 0.6000000000000001)
  Action: -1, Q-value: 0.3592248695560664
  Action: 0, Q-value: 1.2278010734837965
  Action: 1, Q-value: 0.0
State: (-0.75, 0.6000000000000001)
  Action: -1, Q-value: 0.6286579924599407
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-0.5, 0.5)
  Action: -1, Q-value: 2.613529359453671
  Action: 0, Q-value: 1.2073867888354994
  Action: 1, Q-value: 4.216299295612034
State: (-0.5, 0.6000000000000001)
  Action: -1, Q-value: 4.665425171497345
  Action: 0, Q-value: 1.5016216240977704
  Action: 1, Q-value: 0.4285713355696024
State: (-0.5, 0.7000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 1.5125000403720705
  Action: 1, Q-value: 0.0
State: (-0.25, 0.7000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 2.0
  Action: 1, Q-value: 0.0
State: (0.0, 0.7000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 3.7542772624285075
State: (0.0, 0.8)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.8570959023805975
  Action: 1, Q-value: 1.4692984499039343
State: (0.25, 0.8)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.5142736242324873
  Action: 1, Q-value: 0.0
State: (0.5, 0.8)
  Action: -1, Q-value: 0.5861232250819406
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (0.5, 0.7000000000000001)
  Action: -1, Q-value: 1.209363227946103
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (0.5, 0.6000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 1.7239320095940707
  Action: 1, Q-value: 0.25091357185871
State: (0.75, 0.7000000000000001)
  Action: -1, Q-value: 0.21434223052329263
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (1.0, 0.6000000000000001)
  Action: -1, Q-value: 1.091366250251757
  Action: 0, Q-value: 1.3968664859929323
  Action: 1, Q-value: 0.0
State: (1.25, 0.6000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.21638614514468807
  Action: 1, Q-value: 0.712939808327737
State: (1.5, 0.7000000000000001)
  Action: -1, Q-value: 0.7575994305342079
  Action: 0, Q-value: 0.24421463165134558
  Action: 1, Q-value: 0.11442012584571659
State: (2.25, 0.2)
  Action: -1, Q-value: 3.054330761325195
  Action: 0, Q-value: 1.175363304344712
  Action: 1, Q-value: 0.8645648452449379
State: (2.25, 0.1)
  Action: -1, Q-value: 1.9646965999100356
  Action: 0, Q-value: 4.039006816181413
  Action: 1, Q-value: 3.156534262035025
State: (2.5, 0.0)
  Action: -1, Q-value: 4.7979166343469615
  Action: 0, Q-value: 3.7806657473858296
  Action: 1, Q-value: 3.923349564538204
State: (2.5, 0.1)
  Action: -1, Q-value: 4.06374649911394
  Action: 0, Q-value: 3.7277226767092544
  Action: 1, Q-value: 3.6498399950009457
State: (2.5, 0.2)
  Action: -1, Q-value: 3.863809844889863
  Action: 0, Q-value: 2.972720022593746
  Action: 1, Q-value: 1.174325783623291
State: (2.5, -0.1)
  Action: -1, Q-value: 19.11133790638833
  Action: 0, Q-value: 3.84710711845702
  Action: 1, Q-value: 1.9228799455044108
State: (4.0, -0.1)
  Action: -1, Q-value: 0.101763512891757
  Action: 0, Q-value: 0.061735854805538254
  Action: 1, Q-value: 0.32879301373418446
State: (4.0, -0.2)
  Action: -1, Q-value: 2.7969875352567106
  Action: 0, Q-value: 0.4148343141474148
  Action: 1, Q-value: 0.0614952625698338
State: (-2.25, -0.6000000000000001)
  Action: -1, Q-value: 0.1109141432220538
  Action: 0, Q-value: 0.11794320588947659
  Action: 1, Q-value: 0.10887775288701439
State: (-3.0, -0.6000000000000001)
  Action: -1, Q-value: 0.7035511132065366
  Action: 0, Q-value: 0.09070163494555727
  Action: 1, Q-value: 0.2320916908138584
State: (-3.0, -0.5)
  Action: -1, Q-value: 0.7648310099693385
  Action: 0, Q-value: 0.2509426026816538
  Action: 1, Q-value: 0.36778271015828656
State: (-3.25, -0.6000000000000001)
  Action: -1, Q-value: 0.07049281384786053
  Action: 0, Q-value: 0.26716950450371335
  Action: 1, Q-value: 0.0
State: (-3.75, -0.8)
  Action: -1, Q-value: 0.05206358286973694
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-3.75, -0.9)
  Action: -1, Q-value: 0.05972500552129472
  Action: 0, Q-value: 0.04985369102539853
  Action: 1, Q-value: 0.0
State: (-4.0, -0.9)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.12322202392930753
State: (-4.25, -0.9)
  Action: -1, Q-value: -1.927785087558944
  Action: 0, Q-value: 0.0620246891881039
  Action: 1, Q-value: 0.0
State: (-4.5, -1.0)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (2.0, 0.1)
  Action: -1, Q-value: 5.442475435837232
  Action: 0, Q-value: 4.44290636172192
  Action: 1, Q-value: 4.402276032318324
State: (2.25, 0.0)
  Action: -1, Q-value: 0.5421048402802212
  Action: 0, Q-value: 1.5985557862963495
  Action: 1, Q-value: 4.014312268817382
State: (2.25, -0.1)
  Action: -1, Q-value: 23.470101560429363
  Action: 0, Q-value: 0.5796541158302987
  Action: 1, Q-value: 0.23280701403337634
State: (2.25, -0.2)
  Action: -1, Q-value: 37.50008996070028
  Action: 0, Q-value: 0.8728072858740992
  Action: 1, Q-value: 0.9904791418761287
State: (2.0, -0.30000000000000004)
  Action: -1, Q-value: 2.2371304553821503
  Action: 0, Q-value: 2.4896376554833575
  Action: 1, Q-value: 59.67757478584172
State: (2.0, -0.4)
  Action: -1, Q-value: 0.5621314235751966
  Action: 0, Q-value: 2.0205872586640474
  Action: 1, Q-value: 31.39922003780743
State: (2.0, -0.5)
  Action: -1, Q-value: 0.610560590730342
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 1.841324350501294
State: (1.75, -0.4)
  Action: -1, Q-value: 0.5250199357674113
  Action: 0, Q-value: 1.1488134097482725
  Action: 1, Q-value: 0.2557869721339407
State: (1.75, -0.5)
  Action: -1, Q-value: 0.2067413122913837
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 1.0367109828976635
State: (1.25, -0.5)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.3091361700328681
  Action: 1, Q-value: 1.7415380000469636
State: (0.75, -0.1)
  Action: -1, Q-value: 0.6549382150214622
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.5847481086033977
State: (0.75, 0.0)
  Action: -1, Q-value: 0.8320593538475298
  Action: 0, Q-value: 1.061126191027451
  Action: 1, Q-value: 3.6278891077988042
State: (0.75, -0.2)
  Action: -1, Q-value: 103.90923355131055
  Action: 0, Q-value: 2.118099048642279
  Action: 1, Q-value: 0.0
State: (0.75, -0.30000000000000004)
  Action: -1, Q-value: 596.3733235291108
  Action: 0, Q-value: 10.295087937326091
  Action: 1, Q-value: 4.325062352059878
State: (0.0, -0.6000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 2.818389011291033
  Action: 1, Q-value: 15.919268078024043
State: (-0.25, -0.4)
  Action: -1, Q-value: 0.9226137848842471
  Action: 0, Q-value: 1.9848670471842864
  Action: 1, Q-value: 8.101353620318138
State: (-0.75, 0.0)
  Action: -1, Q-value: 4.754373051513685
  Action: 0, Q-value: 2.2377318436417983
  Action: 1, Q-value: 0.9506186664288829
State: (-3.25, -0.4)
  Action: -1, Q-value: 0.19044999923547726
  Action: 0, Q-value: 0.6489356787251035
  Action: 1, Q-value: 0.4241741207990556
State: (-3.25, -0.5)
  Action: -1, Q-value: 0.1517474572559164
  Action: 0, Q-value: 0.11939729676636966
  Action: 1, Q-value: 0.0
State: (-3.5, -0.4)
  Action: -1, Q-value: 0.15998613149660282
  Action: 0, Q-value: 0.4239019875492295
  Action: 1, Q-value: 0.8222559230998108
State: (-3.75, -0.4)
  Action: -1, Q-value: 0.1181506154436036
  Action: 0, Q-value: 0.2370937565111455
  Action: 1, Q-value: 0.5951544018454022
State: (-3.75, -0.30000000000000004)
  Action: -1, Q-value: 0.4265029305024143
  Action: 0, Q-value: 0.7598803282843123
  Action: 1, Q-value: 0.2273744521162064
State: (-4.0, -0.30000000000000004)
  Action: -1, Q-value: 0.19096622367849692
  Action: 0, Q-value: 0.22325126216383295
  Action: 1, Q-value: 0.8267279168376487
State: (-4.0, -0.4)
  Action: -1, Q-value: 0.11781563695944111
  Action: 0, Q-value: 0.1339035823785059
  Action: 1, Q-value: 0.3035265714585019
State: (-4.25, -0.30000000000000004)
  Action: -1, Q-value: 0.21647142764421018
  Action: 0, Q-value: 0.09247931787639689
  Action: 1, Q-value: 0.23003560935090658
State: (-4.25, -0.4)
  Action: -1, Q-value: 0.1709675714131574
  Action: 0, Q-value: 0.07387971585211532
  Action: 1, Q-value: 0.0
State: (-4.25, -0.5)
  Action: -1, Q-value: 0.11043124953849981
  Action: 0, Q-value: 0.193393918806888
  Action: 1, Q-value: 0.045005811707090865
State: (-4.5, -0.6000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-2.0, 0.0)
  Action: -1, Q-value: 1.7832101572758354
  Action: 0, Q-value: 1.7875783408837398
  Action: 1, Q-value: 5.225920869464907
State: (-2.0, 0.1)
  Action: -1, Q-value: 5.497398464487795
  Action: 0, Q-value: 1.5204154491758781
  Action: 1, Q-value: 1.400073785658193
State: (-2.0, -0.1)
  Action: -1, Q-value: 3.4212669292520475
  Action: 0, Q-value: 1.9609509387018433
  Action: 1, Q-value: 1.1424787389409488
State: (-2.0, 0.2)
  Action: -1, Q-value: 0.3624561403626263
  Action: 0, Q-value: 0.25290541949814127
  Action: 1, Q-value: 0.1328018742927399
State: (-1.75, 0.4)
  Action: -1, Q-value: 0.18086634973059912
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-1.0, 0.5)
  Action: -1, Q-value: 0.27420696886832036
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.5993578687056098
State: (-0.25, 0.5)
  Action: -1, Q-value: 14.832384926673452
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 2.234006214513842
State: (0.75, 0.2)
  Action: -1, Q-value: 5.546230559714131
  Action: 0, Q-value: 3.1677409988471172
  Action: 1, Q-value: 0.25718355600355525
State: (0.75, 0.30000000000000004)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 1.5190552806340094
  Action: 1, Q-value: 0.936266897870777
State: (1.75, 0.8)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.10510543986756796
State: (2.0, 0.9)
  Action: -1, Q-value: 0.09629440903763693
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (2.0, 0.8)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.08962255116124401
State: (2.25, 0.9)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.08314866241707461
  Action: 1, Q-value: 0.0
State: (2.5, 0.9)
  Action: -1, Q-value: 0.07267108341082243
  Action: 0, Q-value: 0.07755376875907151
  Action: 1, Q-value: 0.0
State: (2.75, 0.8)
  Action: -1, Q-value: 0.10103722496147319
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (3.25, 0.8)
  Action: -1, Q-value: 0.08050348843388923
  Action: 0, Q-value: 0.17882691125721106
  Action: 1, Q-value: 0.0
State: (2.0, 0.0)
  Action: -1, Q-value: 11.249859267478922
  Action: 0, Q-value: 5.368139762672496
  Action: 1, Q-value: 5.041881065233313
State: (2.0, -0.1)
  Action: -1, Q-value: 4.713373160778741
  Action: 0, Q-value: 4.653027845590769
  Action: 1, Q-value: 40.6058042639834
State: (2.0, -0.2)
  Action: -1, Q-value: 15.040922241263026
  Action: 0, Q-value: 66.06396689533071
  Action: 1, Q-value: 7.029380435894401
State: (3.0, 0.9)
  Action: -1, Q-value: 0.08375503991257123
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (4.5, 1.0)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (2.5, -0.2)
  Action: -1, Q-value: 2.1107205082387837
  Action: 0, Q-value: 4.428947254639555
  Action: 1, Q-value: 14.999060940790852
State: (2.25, -0.30000000000000004)
  Action: -1, Q-value: 45.52383023623631
  Action: 0, Q-value: 0.5129115147586104
  Action: 1, Q-value: 0.37332958424964036
State: (1.25, -0.30000000000000004)
  Action: -1, Q-value: 6.316663981883753
  Action: 0, Q-value: 181.82078496338116
  Action: 1, Q-value: 2.7403930703474284
State: (0.5, -0.2)
  Action: -1, Q-value: 574.1355541739836
  Action: 0, Q-value: 826.5113546619992
  Action: 1, Q-value: 633.3690261011659
State: (-0.5, -0.6000000000000001)
  Action: -1, Q-value: 0.7824916779714364
  Action: 0, Q-value: 1.6990677588981598
  Action: 1, Q-value: 0.6455949904341245
State: (-0.5, -0.7000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.822215676264332
State: (-0.75, -0.7000000000000001)
  Action: -1, Q-value: 0.23381391755011138
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.22365449908162757
State: (-1.0, -0.6000000000000001)
  Action: -1, Q-value: 0.3166228336932229
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.3263062319152601
State: (-1.0, -0.7000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.21002715869805458
State: (-1.25, -0.7000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.14295837449056545
  Action: 1, Q-value: 0.0
State: (-1.5, -0.7000000000000001)
  Action: -1, Q-value: 0.4554058076272641
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-1.5, -0.8)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.2302189980792082
  Action: 1, Q-value: 0.18859922224482994
State: (-1.75, -0.8)
  Action: -1, Q-value: 0.10300991287555877
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.242323822299709
State: (-2.0, -0.8)
  Action: -1, Q-value: 0.10318524871012652
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-2.5, -1.0)
  Action: -1, Q-value: 0.1354452774073071
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-2.5, -1.1)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.1968084994490232
  Action: 1, Q-value: 0.0
State: (-3.0, -1.1)
  Action: -1, Q-value: 0.19524668793455768
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-3.0, -1.2000000000000002)
  Action: -1, Q-value: 0.06080431612512345
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.21288381500350217
State: (-3.25, -1.3)
  Action: -1, Q-value: 0.05653893952177741
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-3.5, -1.4000000000000001)
  Action: -1, Q-value: 0.05258262023511774
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-3.75, -1.5)
  Action: -1, Q-value: 0.05710276092346547
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-4.25, -1.7000000000000002)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (4.25, -0.2)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.15728927359715597
  Action: 1, Q-value: 0.12869053201292596
State: (2.75, -0.4)
  Action: -1, Q-value: 1.0394304006129804
  Action: 0, Q-value: 0.15288974709153105
  Action: 1, Q-value: 0.21306662362244025
State: (-0.75, 0.2)
  Action: -1, Q-value: 85.17165985068831
  Action: 0, Q-value: 577.9275958047756
  Action: 1, Q-value: 236.63819652807732
State: (-0.75, 0.4)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.6504798843108238
State: (-0.25, 0.6000000000000001)
  Action: -1, Q-value: 2.132238412462456
  Action: 0, Q-value: 2.3292699793671328
  Action: 1, Q-value: 1.6456075184929764
State: (0.0, 0.6000000000000001)
  Action: -1, Q-value: 4.248573724638916
  Action: 0, Q-value: 5.0355978864591275
  Action: 1, Q-value: 0.0
State: (0.25, 0.6000000000000001)
  Action: -1, Q-value: 0.6631112791115908
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 1.228980857612412
State: (1.0, 0.7000000000000001)
  Action: -1, Q-value: 0.7967868892056351
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.20997136146639028
State: (3.25, 0.4)
  Action: -1, Q-value: 0.5548154410068481
  Action: 0, Q-value: 0.0978765807274069
  Action: 1, Q-value: 0.08380544309191203
State: (3.75, -0.30000000000000004)
  Action: -1, Q-value: 0.05420614465010633
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.33852792894090866
State: (1.0, -0.6000000000000001)
  Action: -1, Q-value: 0.2780259647988604
  Action: 0, Q-value: 0.4230679873047774
  Action: 1, Q-value: 9.394344659418673
State: (0.75, -0.7000000000000001)
  Action: -1, Q-value: 0.47247036873920606
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (0.0, -1.1)
  Action: -1, Q-value: 1.2856047827096528
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (0.0, -1.2000000000000002)
  Action: -1, Q-value: 0.5142719029102091
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-0.5, -1.3)
  Action: -1, Q-value: 0.36553908567154164
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.3568761896978989
State: (-1.0, -1.2000000000000002)
  Action: -1, Q-value: 0.15135426570436114
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-1.5, -1.3)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.2350454570701371
State: (-1.5, -1.2000000000000002)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.1981552541822117
  Action: 1, Q-value: 0.0
State: (-1.75, -1.2000000000000002)
  Action: -1, Q-value: 0.196214335147033
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-2.0, -1.3)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.17456844627193335
State: (-2.25, -1.2000000000000002)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0793322518762562
  Action: 1, Q-value: 0.09447515251947793
State: (-2.5, -1.2000000000000002)
  Action: -1, Q-value: 0.07267101168582202
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-2.75, -1.3)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.07756918261565159
State: (-3.75, -1.2000000000000002)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.05130464802289339
  Action: 1, Q-value: 0.05913569042621845
State: (-4.0, -1.2000000000000002)
  Action: -1, Q-value: 0.04848584408121818
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (3.75, 0.5)
  Action: -1, Q-value: 0.3561328599445193
  Action: 0, Q-value: 0.1515692249687947
  Action: 1, Q-value: 0.0
State: (-1.25, 0.1)
  Action: -1, Q-value: 0.15833013691142156
  Action: 0, Q-value: 0.8227782804206438
  Action: 1, Q-value: 58.76220649957322
State: (-1.25, 0.2)
  Action: -1, Q-value: 4.815510094446121
  Action: 0, Q-value: 21.810419528539946
  Action: 1, Q-value: 184.47713682874516
State: (-1.0, 0.4)
  Action: -1, Q-value: 50.717601004899116
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.43904196277963736
State: (-0.5, 0.2)
  Action: -1, Q-value: 304.0196922651984
  Action: 0, Q-value: 248.83132881496172
  Action: 1, Q-value: 736.1185215062105
State: (0.25, 0.5)
  Action: -1, Q-value: 0.6793488104456211
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.7980598969977399
State: (3.25, 0.0)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 1739.3326500913806
State: (3.25, -0.5)
  Action: -1, Q-value: 0.1250453300957146
  Action: 0, Q-value: 0.1991810508032493
  Action: 1, Q-value: 0.0
State: (2.75, -0.5)
  Action: -1, Q-value: 0.39387286589951237
  Action: 0, Q-value: 1.3974839010259175
  Action: 1, Q-value: 0.0929439832746124
State: (0.75, -0.6000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.3214513200296927
  Action: 1, Q-value: 0.8464191822704783
State: (-0.25, -0.6000000000000001)
  Action: -1, Q-value: 0.6403328270310832
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-0.75, -0.6000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.2962406006571718
  Action: 1, Q-value: 0.43455218947394036
State: (-1.0, -0.5)
  Action: -1, Q-value: 0.34592336024909454
  Action: 0, Q-value: 0.8632684383374458
  Action: 1, Q-value: 1.5024472354774
State: (-3.75, -0.2)
  Action: -1, Q-value: 0.3120016235174218
  Action: 0, Q-value: 0.08242668365580237
  Action: 1, Q-value: 0.052321767968642956
State: (2.0, -0.6000000000000001)
  Action: -1, Q-value: 0.13318768850404517
  Action: 0, Q-value: 0.23143842826474234
  Action: 1, Q-value: 0.879621532697324
State: (1.25, 0.0)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.45618597337856626
  Action: 1, Q-value: 0.47181685928452854
State: (1.25, 0.1)
  Action: -1, Q-value: 0.20918542620405917
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 1.6577138241760514
State: (4.5, 0.7000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (3.25, -0.1)
  Action: -1, Q-value: 3.5920777989964052
  Action: 0, Q-value: 0.0950042915627559
  Action: 1, Q-value: 0.07974911497292683
State: (4.75, -0.1)
  Action: -1, Q-value: 0.13108679688133984
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (2.25, -0.4)
  Action: -1, Q-value: 0.45977503632063305
  Action: 0, Q-value: 0.50295889840398
  Action: 1, Q-value: 53.33767282932874
State: (1.75, -0.6000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.12555526833926978
  Action: 1, Q-value: 0.0
State: (1.5, -0.6000000000000001)
  Action: -1, Q-value: 0.18976678075855077
  Action: 0, Q-value: 0.23717458805338848
  Action: 1, Q-value: 0.45941927847981695
State: (1.5, -0.5)
  Action: -1, Q-value: 0.5807583964675498
  Action: 0, Q-value: 1.0359012427537855
  Action: 1, Q-value: 0.0
State: (-0.75, -0.30000000000000004)
  Action: -1, Q-value: 0.6393654304774654
  Action: 0, Q-value: 2.6650003863837997
  Action: 1, Q-value: 0.4659570351271161
State: (-1.5, -0.5)
  Action: -1, Q-value: 0.11570215702054557
  Action: 0, Q-value: 0.5041230807882966
  Action: 1, Q-value: 0.973782094279175
State: (-1.75, -0.6000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.14133257785888542
  Action: 1, Q-value: 0.1992994580728656
State: (-2.75, -0.7000000000000001)
  Action: -1, Q-value: 0.08308366883434737
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.20358999127699876
State: (-3.25, -1.0)
  Action: -1, Q-value: 0.05653892263877355
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.05684517434875176
State: (-3.5, -0.9)
  Action: -1, Q-value: 0.11584581282821885
  Action: 0, Q-value: 0.1687536001546341
  Action: 1, Q-value: 0.0
State: (-3.75, -1.0)
  Action: -1, Q-value: 0.15495407462586236
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-4.0, -1.0)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.04677982285603931
  Action: 1, Q-value: 0.061339664941286266
State: (-4.25, -1.0)
  Action: -1, Q-value: 0.14204441292551614
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-4.5, -1.1)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (0.25, 0.0)
  Action: -1, Q-value: 1185.2079264676697
  Action: 0, Q-value: 137.45389130896808
  Action: 1, Q-value: 133.68517393492698
State: (0.25, -0.1)
  Action: -1, Q-value: 1409.4409988848392
  Action: 0, Q-value: 816.9430795104445
  Action: 1, Q-value: 442.8245223826272
State: (-0.25, 0.0)
  Action: -1, Q-value: 10.61355580321288
  Action: 0, Q-value: 5.659504699769056
  Action: 1, Q-value: 994.8224011484921
State: (-0.5, 0.1)
  Action: -1, Q-value: 52.29194287087159
  Action: 0, Q-value: 90.16314170020523
  Action: 1, Q-value: 476.27851523437107
State: (-3.5, -0.30000000000000004)
  Action: -1, Q-value: 0.8075723014324064
  Action: 0, Q-value: 0.10801570223470532
  Action: 1, Q-value: 0.2799549681520227
State: (-1.25, -0.2)
  Action: -1, Q-value: 0.27737967622120535
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.9252634955184802
State: (-2.25, -0.30000000000000004)
  Action: -1, Q-value: 1.5483586039283217
  Action: 0, Q-value: 0.11599429576858436
  Action: 1, Q-value: 0.500325784371881
State: (-2.25, -0.2)
  Action: -1, Q-value: 1.2423270637606123
  Action: 0, Q-value: 0.4075772146888864
  Action: 1, Q-value: 0.32002978665728765
State: (-2.5, -0.2)
  Action: -1, Q-value: 0.772509837445201
  Action: 0, Q-value: 0.43704109017627985
  Action: 1, Q-value: 3.307116596026061
State: (-2.5, 0.0)
  Action: -1, Q-value: 0.8117122319194361
  Action: 0, Q-value: 2.9745679425944447
  Action: 1, Q-value: 0.5452940374704468
State: (-2.5, 0.1)
  Action: -1, Q-value: 1.1493460760336447
  Action: 0, Q-value: 0.16717789935961133
  Action: 1, Q-value: 0.0
State: (-2.5, -0.30000000000000004)
  Action: -1, Q-value: 0.5098699449153858
  Action: 0, Q-value: 1.3470891642792584
  Action: 1, Q-value: 0.510160050042975
State: (-2.75, -0.30000000000000004)
  Action: -1, Q-value: 0.4388425823769103
  Action: 0, Q-value: 0.19603955884841687
  Action: 1, Q-value: 0.15714499118048203
State: (-2.75, -0.4)
  Action: -1, Q-value: 0.3236048755926769
  Action: 0, Q-value: 0.07069511974230717
  Action: 1, Q-value: 107.62756823570508
State: (-3.0, -0.30000000000000004)
  Action: -1, Q-value: 0.5231274053482832
  Action: 0, Q-value: 0.9120180803870943
  Action: 1, Q-value: 0.3282825941305871
State: (-3.0, -0.2)
  Action: -1, Q-value: 0.4349716193387886
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-3.25, -0.2)
  Action: -1, Q-value: 0.06225262835740151
  Action: 0, Q-value: 0.5337680818724853
  Action: 1, Q-value: 0.05908794646317825
State: (-3.25, -0.30000000000000004)
  Action: -1, Q-value: 0.5675115304684537
  Action: 0, Q-value: 0.1675685664546928
  Action: 1, Q-value: 0.0
State: (-4.0, -0.2)
  Action: -1, Q-value: 0.29225702387571173
  Action: 0, Q-value: 1.0221686809593864
  Action: 1, Q-value: 0.39899698692994107
State: (-4.25, -0.2)
  Action: -1, Q-value: 0.28576418713734275
  Action: 0, Q-value: 1.1771846207217056
  Action: 1, Q-value: 0.3021420355912364
State: (-4.25, -0.1)
  Action: -1, Q-value: 0.9591428114407583
  Action: 0, Q-value: 0.05694241991294711
  Action: 1, Q-value: 0.0
State: (0.25, -0.2)
  Action: -1, Q-value: 783.415202701256
  Action: 0, Q-value: 1407.9176139197748
  Action: 1, Q-value: 1130.609324974477
State: (-1.25, -0.1)
  Action: -1, Q-value: 0.7287228599076032
  Action: 0, Q-value: 0.22618257181116508
  Action: 1, Q-value: 2.779804888710106
State: (-1.25, 0.30000000000000004)
  Action: -1, Q-value: 90.70250371062428
  Action: 0, Q-value: 256.88733597037174
  Action: 1, Q-value: 1.8480942531952183
State: (-0.75, 0.5)
  Action: -1, Q-value: 4.175471798891701
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-0.25, 0.2)
  Action: -1, Q-value: 657.5562183797992
  Action: 0, Q-value: 524.8913969542326
  Action: 1, Q-value: 1414.797029924094
State: (-0.25, 0.30000000000000004)
  Action: -1, Q-value: 1595.099000861881
  Action: 0, Q-value: 165.1542792713201
  Action: 1, Q-value: 3.5994267729005953
State: (2.75, -0.7000000000000001)
  Action: -1, Q-value: 0.1703955058839794
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (2.5, -0.8)
  Action: -1, Q-value: 0.09040518623399289
  Action: 0, Q-value: 0.3880179196601617
  Action: 1, Q-value: 0.0
State: (1.25, -0.6000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.6946147356049889
  Action: 1, Q-value: 0.0
State: (0.75, 0.1)
  Action: -1, Q-value: 2.6115262238926706
  Action: 0, Q-value: 6.222227183589938
  Action: 1, Q-value: 3.8864364246324103
State: (-2.75, -0.2)
  Action: -1, Q-value: 0.23974856007042467
  Action: 0, Q-value: 0.17138013547908942
  Action: 1, Q-value: 0.18389610148127716
State: (-3.0, -0.4)
  Action: -1, Q-value: 0.09372304549732768
  Action: 0, Q-value: 0.13791978987792122
  Action: 1, Q-value: 0.7380035488982658
State: (-4.5, -0.4)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-1.25, 0.0)
  Action: -1, Q-value: 1.4044875350027302
  Action: 0, Q-value: 0.15833020353046007
  Action: 1, Q-value: 0.0
State: (-4.5, -0.5)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (0.25, 0.1)
  Action: -1, Q-value: 725.4293361080206
  Action: 0, Q-value: 15.988636982144273
  Action: 1, Q-value: 9.94400325071946
State: (-4.0, -0.5)
  Action: -1, Q-value: 0.08013552946624969
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0694991960680093
State: (0.25, -0.30000000000000004)
  Action: -1, Q-value: 6.804097933486905
  Action: 0, Q-value: 1157.1731142719593
  Action: 1, Q-value: 0.0
State: (-4.0, -0.8)
  Action: -1, Q-value: 0.10906648362328683
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.2440552858723294
State: (1.25, -0.8)
  Action: -1, Q-value: 0.19054452181445689
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (1.0, -0.9)
  Action: -1, Q-value: 0.2286221161696989
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (1.0, -1.0)
  Action: -1, Q-value: 0.293906932400725
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (0.5, -1.1)
  Action: -1, Q-value: 0.42857356836940563
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (0.5, -1.2000000000000002)
  Action: -1, Q-value: 1.9712965270748182
  Action: 0, Q-value: 0.668552599842571
  Action: 1, Q-value: 0.0
State: (0.25, -1.3)
  Action: -1, Q-value: 2.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (0.0, -1.4000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 1.154110327924606
State: (-0.25, -1.3)
  Action: -1, Q-value: 0.3673582585361806
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-0.5, -1.4000000000000001)
  Action: -1, Q-value: 0.4521018409269911
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-1.0, -1.5)
  Action: -1, Q-value: 0.3384990639253618
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-1.0, -1.6)
  Action: -1, Q-value: 0.2670892024769842
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-1.5, -1.7000000000000002)
  Action: -1, Q-value: 0.21583050382289853
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-1.75, -1.8)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.09540607133410604
  Action: 1, Q-value: 0.0
State: (-2.0, -1.8)
  Action: -1, Q-value: 0.15748108868046534
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-2.5, -1.9000000000000001)
  Action: -1, Q-value: 0.07117881417969125
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-2.75, -2.0)
  Action: -1, Q-value: 0.06262576783929517
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-3.25, -2.1)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.05564028896808376
State: (-3.5, -2.0)
  Action: -1, Q-value: 0.0503277630521402
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-4.0, -2.1)
  Action: -1, Q-value: 0.04577633490400329
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-4.25, -2.2)
  Action: -1, Q-value: -2.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-4.75, -2.3000000000000003)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-0.75, 0.1)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 218.34510056392162
State: (4.0, 0.9)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.15538072972921052
State: (-1.25, -0.5)
  Action: -1, Q-value: 0.29326790128629954
  Action: 0, Q-value: 0.264758782385125
  Action: 1, Q-value: 0.0
State: (-1.5, -0.6000000000000001)
  Action: -1, Q-value: 0.1589031712923137
  Action: 0, Q-value: 0.3715123700061321
  Action: 1, Q-value: 0.0
State: (-2.25, -0.7000000000000001)
  Action: -1, Q-value: 0.09898633457650323
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-3.5, -1.3)
  Action: -1, Q-value: 0.05206360610593616
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-3.75, -1.4000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.04870835114186128
  Action: 1, Q-value: 0.0
State: (-4.0, -1.4000000000000001)
  Action: -1, Q-value: 0.11366560594810984
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-4.25, -1.5)
  Action: -1, Q-value: -2.0
  Action: 0, Q-value: -2.0
  Action: 1, Q-value: -2.0
State: (-4.75, -1.6)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-2.25, -0.1)
  Action: -1, Q-value: 0.39196588359391704
  Action: 0, Q-value: 0.3901058491851682
  Action: 1, Q-value: 1.8627671277508848
State: (-2.25, 0.0)
  Action: -1, Q-value: 0.9389887473180286
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.19494799123510942
State: (0.5, -0.6000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 1.0271931826474958
  Action: 1, Q-value: 12.360359268545853
State: (-2.75, -0.1)
  Action: -1, Q-value: 0.3018361089626084
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-3.5, -0.1)
  Action: -1, Q-value: 0.38310435724665504
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-3.5, -0.2)
  Action: -1, Q-value: 0.6442017166763432
  Action: 0, Q-value: 0.16128871243598453
  Action: 1, Q-value: 0.16230780146864732
State: (-4.5, -0.7000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (4.0, 0.8)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.06300063440853997
State: (4.75, -0.30000000000000004)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.13925249817953805
  Action: 1, Q-value: 0.0
State: (0.25, -0.6000000000000001)
  Action: -1, Q-value: 1.2856315034972101
  Action: 0, Q-value: 2.2876792490324944
  Action: 1, Q-value: 0.0
State: (0.0, -0.7000000000000001)
  Action: -1, Q-value: 2.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (0.0, -0.8)
  Action: -1, Q-value: 0.6856754170593529
  Action: 0, Q-value: 1.469253531732872
  Action: 1, Q-value: 0.0
State: (-0.25, -0.9)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.47266093609115034
  Action: 1, Q-value: 0.0
State: (-1.25, -1.2000000000000002)
  Action: -1, Q-value: 0.15832508948116564
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-3.75, -1.3)
  Action: -1, Q-value: 0.1115664354870666
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-4.5, -1.5)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-4.0, -0.1)
  Action: -1, Q-value: 0.3289006043452347
  Action: 0, Q-value: 0.1314845360666982
  Action: 1, Q-value: 0.08772168027866178
State: (0.0, 0.9)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.6428319920217431
State: (0.5, 1.0)
  Action: -1, Q-value: 0.3956112658349814
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (0.5, 0.9)
  Action: -1, Q-value: 0.3600316835002656
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (1.0, 0.8)
  Action: -1, Q-value: 0.3061465572088775
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (2.5, 0.7000000000000001)
  Action: -1, Q-value: 0.1299358853834945
  Action: 0, Q-value: 0.0811952157930753
  Action: 1, Q-value: 0.0
State: (3.75, -0.4)
  Action: -1, Q-value: 63.32513478469962
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.19682664973106198
State: (3.25, -0.6000000000000001)
  Action: -1, Q-value: 0.0653713420444991
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (3.0, -0.7000000000000001)
  Action: -1, Q-value: 0.06837337755925474
  Action: 0, Q-value: 0.08766128817859763
  Action: 1, Q-value: 0.0
State: (3.0, -0.8)
  Action: -1, Q-value: 0.07216671811511104
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (2.75, -0.9)
  Action: -1, Q-value: 0.07697875192347178
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (2.5, -1.0)
  Action: -1, Q-value: 0.08314865246333321
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (2.5, -1.1)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.09120177165340138
State: (2.25, -1.0)
  Action: -1, Q-value: 0.10001988923900612
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (2.0, -1.1)
  Action: -1, Q-value: 0.21212391618753806
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (1.75, -1.2000000000000002)
  Action: -1, Q-value: 0.12868751409952603
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (1.5, -1.3)
  Action: -1, Q-value: 0.297300232825113
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (1.25, -1.4000000000000001)
  Action: -1, Q-value: 0.19413627477939835
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (1.0, -1.5)
  Action: -1, Q-value: 0.27071304410517916
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (0.75, -1.6)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.46752819655177635
  Action: 1, Q-value: 0.0
State: (0.5, -1.6)
  Action: -1, Q-value: 1.7141695807562252
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (0.0, -1.7000000000000002)
  Action: -1, Q-value: 0.935000120182669
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-0.25, -1.8)
  Action: -1, Q-value: 0.354693889672396
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-0.5, -1.9000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.21434154560595006
  Action: 1, Q-value: 0.0
State: (-1.0, -1.9000000000000001)
  Action: -1, Q-value: 0.15361017517270412
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-1.25, -2.0)
  Action: -1, Q-value: 0.11835530433645369
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-1.75, -2.1)
  Action: -1, Q-value: 0.09540607099093018
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-2.0, -2.2)
  Action: -1, Q-value: 0.07933221789207494
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-2.5, -2.3000000000000003)
  Action: -1, Q-value: 0.0674873617997213
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-3.0, -2.4000000000000004)
  Action: -1, Q-value: 0.05842883232205845
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-3.5, -2.5)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.05130465879033409
State: (-4.0, -2.4000000000000004)
  Action: -1, Q-value: 0.04597336709500743
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-4.25, -2.5)
  Action: -1, Q-value: -2.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-4.75, -2.6)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (3.75, 0.8)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.08157304732552775
State: (-3.25, 0.0)
  Action: -1, Q-value: 1799.5612225616287
  Action: 0, Q-value: 381.23654430400444
  Action: 1, Q-value: 0.0
State: (-1.25, -0.6000000000000001)
  Action: -1, Q-value: 0.19196416897649432
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-2.0, -0.9)
  Action: -1, Q-value: 0.11080719206862821
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-3.5, -1.1)
  Action: -1, Q-value: 0.06261645996100076
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0639633570254117
State: (-4.25, -1.1)
  Action: -1, Q-value: -1.9569149599397682
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: -2.0
State: (-0.25, 0.1)
  Action: -1, Q-value: 5.026677685596157
  Action: 0, Q-value: 1486.6600166945113
  Action: 1, Q-value: 837.3768547580204
State: (-2.25, 0.1)
  Action: -1, Q-value: 1.1060281972200006
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-4.25, 0.0)
  Action: -1, Q-value: 0.11124800182882824
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 3.3864903687214643
State: (-1.0, -0.8)
  Action: -1, Q-value: 0.19786616621575734
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-1.0, -0.9)
  Action: -1, Q-value: 0.23808959386186773
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-1.5, -1.1)
  Action: -1, Q-value: 0.1454881130646286
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-2.0, -1.2000000000000002)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.09719920613830538
State: (-2.0, -1.1)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.08809748229142032
  Action: 1, Q-value: 0.0
State: (-2.25, -1.1)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.09421815533303213
State: (-2.75, -1.1)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0927304453430242
  Action: 1, Q-value: 0.0
State: (-4.5, -1.4000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-3.5, -1.0)
  Action: -1, Q-value: 0.06750703443466433
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-4.5, -1.2000000000000002)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-4.0, 0.0)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.5815902276140056
  Action: 1, Q-value: 0.15715028891707283
State: (-4.0, 0.1)
  Action: -1, Q-value: 0.1404352734216375
  Action: 0, Q-value: 0.2031445885614985
  Action: 1, Q-value: 0.10753143937299722
State: (-4.0, 0.2)
  Action: -1, Q-value: 0.08762278737819636
  Action: 0, Q-value: 0.051808019242424556
  Action: 1, Q-value: 0.0
State: (-3.75, 0.2)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.10676643719951409
State: (-3.75, 0.30000000000000004)
  Action: -1, Q-value: 0.10537155136380522
  Action: 0, Q-value: 0.06574229851838775
  Action: 1, Q-value: 0.0
State: (-3.5, 0.2)
  Action: -1, Q-value: 0.05593655439380129
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-3.5, 0.1)
  Action: -1, Q-value: 0.05623609602020296
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-3.5, 0.0)
  Action: -1, Q-value: 0.08858533164857796
  Action: 0, Q-value: 0.05623608554219237
  Action: 1, Q-value: 0.0
State: (-3.75, -0.1)
  Action: -1, Q-value: 0.08917360824135753
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-4.5, -0.2)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (4.5, 0.1)
  Action: -1, Q-value: 0.3743650868431006
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (2.25, -0.9)
  Action: -1, Q-value: 0.09812157497154207
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (2.0, -1.0)
  Action: -1, Q-value: 0.12856338054379843
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (1.5, -1.2000000000000002)
  Action: -1, Q-value: 0.1706084162601016
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (1.0, -1.4000000000000001)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.2286222443250089
State: (1.0, -1.3)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.4757282942891271
State: (0.0, -1.3)
  Action: -1, Q-value: 2.12342133945628
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-2.0, -1.9000000000000001)
  Action: -1, Q-value: 0.0787302802730999
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-2.5, -2.0)
  Action: -1, Q-value: 0.06837331770312143
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-3.0, -2.1)
  Action: -1, Q-value: 0.060105662015183
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-3.25, -2.2)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.053381600109741946
  Action: 1, Q-value: 0.0
State: (-3.75, -2.2)
  Action: -1, Q-value: 0.04804718793389829
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-4.25, -2.3000000000000003)
  Action: -1, Q-value: -2.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-4.75, -2.4000000000000004)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-3.25, -0.1)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 354.58866138382587
  Action: 1, Q-value: 399.7864281195187
State: (4.25, -0.30000000000000004)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.9155173927839005
State: (-3.25, 0.1)
  Action: -1, Q-value: 384.8187247144875
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (2.0, 0.7000000000000001)
  Action: -1, Q-value: 0.17898589538150725
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (-4.5, -0.1)
  Action: -1, Q-value: 0.0
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (5.25, 0.0)
  Action: -1, Q-value: 0.9709040366503789
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (5.25, 0.1)
  Action: -1, Q-value: 0.3885845505838601
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (4.0, -0.30000000000000004)
  Action: -1, Q-value: 1.3444398714914163
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (4.5, 0.0)
  Action: -1, Q-value: 1.2547970086904128
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0
State: (5.0, -0.2)
  Action: -1, Q-value: 0.5055307842796288
  Action: 0, Q-value: 0.0
  Action: 1, Q-value: 0.0


"""
parse_q_table_to_python(input_text)
