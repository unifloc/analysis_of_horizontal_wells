import os
from ecl.summary import EclSum
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import subprocess
from plotly.offline import iplot
from jinja2 import Environment, FileSystemLoader
from ecl2df import grid, EclFiles
import plotly.express as px
import math as m
from scipy.interpolate import interp1d

"""
@alexeyvodopyan
@sevrn
"""

# метод для очистки результатов расчетов
def clear_folders():
    subprocess.call("rm -f model_folder/*", shell=True)
    subprocess.call("rm -f csv_folder/*", shell=True)


class ModelGenerator:
    """
    Класс для расчета модели
    """

    def __init__(self, start_date="1 'SEP' 2020", mounths=2, days=30, nx=20, ny=20, nz=5, dx=100, dy=100, dz=5, por=0.3, permx=100,
                 permy=100, permz=10, prod_names=None, prod_xs=None, prod_ys=None, prod_z1s=None, prod_z2s=None, prod_q_oil=None,
                 inj_names=None, inj_xs=None, inj_ys=None, inj_z1s=None, inj_z2s=None, inj_bhp=None, skin=None, oil_den=860, wat_den=1010, gas_den=0.9,
                 p_depth=2500, p_init=250, o_w_contact=2600, pc_woc=0, g_o_contact=2450, pc_goc=0, tops_depth=2500, rock_compr=1.5E-005,
                 rezim='ORAT', prod_bhp=None, horizontal=False, y_stop=None, only_prod=False,
                 lgr=False, lx=None, ly=None, cells_cy=None, cells_v=None, cells_cx=None,
                 upr_rezim_water=False, upr_rezim_gas=False, rw=None, template=1, neogr=False,
                 grp=False, nz_grp=1, xs_start_grp=1, xs_stop_grp=2, ys_grp=1, k_grp=100, roughness=False):
        # продолжительность расчета
        self.start_date = f'{start_date}'
        self.tstep = f'{mounths}*{days}'

        # размеры модели
        self.dimens = f'{nx} {ny} {nz}'
        self.dx = f'{nx*ny*nz}*{dx} /'
        self.dy = f'{nx*ny*nz}*{dy} /'
        self.dz = f'{nx*ny*nz}*{dz}'
        if template == 2 or template == 4: 
            self.dz = f'DZ {dz} / \n'
            self.dz += '/'
        self.top_box = ''
        if template == 2 or template == 4:
            self.top_box = 'BOX \n'
            self.top_box += f'1 {nx} 1 {ny} 1 1 /'
        self.tops_depth = f'{nx*ny}*{tops_depth} '

        # LGR
        if lgr == True:
            if template == 1 or template == 3:
                # формируем измельченную сетку по x
                dx_lgr = self.setcas(nx, lx, cells_cx, cells_v)
                self.dx = str(dx_lgr[0]) + ' \n'
                for i in range(2, nx + 1):    
                    self.dx += str(dx_lgr[i-1]) + ' \n'
                self.dx = self.dx*ny
                self.dx += '/\n'

                # формируем измельченную сетку по y
                dy_lgr = self.setcas(ny, ly, cells_cy, cells_v)
                self.dy = ''
                for i in range(1, ny + 1):    
                    dim = str(dy_lgr[i-1]) + ' \n'
                    self.dy += dim*nx
                self.dy += '/\n'
            elif template == 2 or template == 4:
                # формируем измельченную сетку по x
                dx_lgr = self.setcas(nx, lx, cells_cx, cells_v)
                self.dx = 'DX ' + str(dx_lgr[0]) + ' 1 1 /\n'
                for i in range(2, nx + 1):    
                    self.dx += 'DX ' + str(dx_lgr[i-1]) + f' {i} {i} /\n'
                self.dx += '/\n'

                # формируем измельченную сетку по y
                dy_lgr = self.setcas(ny, ly, cells_cy, cells_v)
                self.dy = ''
                for i in range(1, ny + 1):    
                    self.dy += 'DY ' + str(dy_lgr[i-1]) + f' 2* {i} {i} /\n'
                self.dy += '/\n'

        # физические свойства
        self.por = f'{nx*ny*nz}*{por}'
        if template == 1 or template == 3:
            self.permx = f'{nx*ny*nz}*{permx}'
            self.permy = f'{nx*ny*nz}*{permy}'
            self.permz = f'{nx*ny*nz}*{permz}'
        elif template == 2 or template == 4:
            self.permx = f'PERMX {permx} 6*/ \n'
            self.permy = f'PERMY {permy} 6*/ \n'
            self.permz = f'PERMZ {permz} 6*/ \n'


        # EQUILIBRIUM DATA
        self.equil = f'{p_depth} {p_init} {o_w_contact} {pc_woc} {g_o_contact} {pc_goc} 1 1* 1*  /'

        # свойства продукции
        self.density = f'{oil_den} {wat_den} {gas_den} /'

        # свойства породы
        self.rock = f'{p_init} {rock_compr}'

        # индикаторы режимов (в разработке)
        self.only_prod = only_prod # оставляет только добывающие скважины
        self.upr_rezim_water = upr_rezim_water # умножает на 1000 пористость последнего пропласта (предварительно необходимо установить ВНК)
        self.upr_rezim_gas = upr_rezim_gas # умножает на 100 пористость первого пропластка (предварителньо необходимо установить ГНК)
        self.neogr = neogr

        # умножение порового объема (неограниченный пласт)
        self.poro_box = ''
        if template == 2 or template == 4: 
            self.poro_box = 'BOX \n'
            self.poro_box += f'1 {nx} 1 {ny} 1 {nz} /'
        self.por = ''
        if self.upr_rezim_water and self.upr_rezim_gas and not self.neogr:
            self.por = str(nx*ny) + '*1000' + ' ' + str(nx*ny*(nz-2)) + '*' + str(por) + ' ' + str(nx*ny) + '*1000' +  ' /'
        elif self.upr_rezim_gas and not self.upr_rezim_water and not self.neogr:
            self.por = str(nx*ny) + '*1000' + ' ' + str(nx*ny*(nz-1)) + '*' + str(por) +  ' /'
        elif self.upr_rezim_water and not self.upr_rezim_gas and not self.neogr:
            self.por = str(nx*ny*(nz-1)) + '*' + str(por) + ' ' + str(nx*ny) + '*1000' +  ' /'
        elif self.neogr and self.upr_rezim_water and self.upr_rezim_gas:
            for j in range(0, nz):
                val = por 
                if j == 0: val = 1000
                if j == nz-1: val = 1000
                self.por += str(nx+1) + '*' + str(1000) + ' \n'
                for i in range(0, ny-3):
                    self.por += str(nx-2) + '*' + str(val) + ' \n'
                    self.por += str(2) + '*' + str(1000) + ' \n'
                self.por += str(nx-2) + '*' + str(val) + ' \n'
                self.por += str(nx+1) + '*' + str(1000) + ' \n'
        elif self.neogr and self.upr_rezim_water and not self.upr_rezim_gas:
            for j in range(0, nz):
                val = por 
                if j == nz-1: val = 1000
                self.por += str(nx+1) + '*' + str(1000) + ' \n'
                for i in range(0, ny-3):
                    self.por += str(nx-2) + '*' + str(val) + ' \n'
                    self.por += str(2) + '*' + str(1000) + ' \n'
                self.por += str(nx-2) + '*' + str(val) + ' \n'
                self.por += str(nx+1) + '*' + str(1000) + ' \n'
        elif self.neogr and not self.upr_rezim_water and self.upr_rezim_gas:
            for j in range(0, nz):
                val = por 
                if j == 0: val = 1000
                self.por += str(nx+1) + '*' + str(1000) + ' \n'
                for i in range(0, ny-3):
                    self.por += str(nx-2) + '*' + str(val) + ' \n'
                    self.por += str(2) + '*' + str(1000) + ' \n'
                self.por += str(nx-2) + '*' + str(val) + ' \n'
                self.por += str(nx+1) + '*' + str(1000) + ' \n'
        elif self.neogr and not self.upr_rezim_water and not self.upr_rezim_gas:
            for j in range(0, nz):
                self.por += str(nx+1) + '*' + str(1000) + ' \n'
                for i in range(0, ny-3):
                    self.por += str(nx-2) + '*' + str(por) + ' \n'
                    self.por += str(2) + '*' + str(1000) + ' \n'
                self.por += str(nx-2) + '*' + str(por) + ' \n'
                self.por += str(nx+1) + '*' + str(1000) + ' \n'
        elif not self.neogr and not self.upr_rezim_water and not self.upr_rezim_gas:
            self.por += str(nx*ny*nz) + '*' + str(por) +  ' /'

        # SCHEDULE секция
        if not only_prod:
            all_well_names = prod_names + inj_names
            all_well_xs = prod_xs + inj_xs
            all_well_ys = prod_ys + inj_ys
            all_well_z1s = prod_z1s + inj_z1s
            all_well_z2s = prod_z2s + inj_z2s
            all_well_fluid = ['OIL' for _ in range(len(prod_names))] + ['WAT' for _ in range(len(inj_names))]
        else:
            all_well_names = prod_names
            all_well_xs = prod_xs
            all_well_ys = prod_ys 
            all_well_z1s = prod_z1s
            all_well_z2s = prod_z2s
            all_well_fluid = ['OIL' for _ in range(len(prod_names))]
        
        self.welspecs = ''
        self.wconprod = ''
        self.compdat = ''
        for name, x, y, fluid in zip(all_well_names, all_well_xs,
                                             all_well_ys, all_well_fluid):
            if template == 2 or template == 4: name = f'"{name}"'
            self.welspecs += name + ' G1 ' + str(x) + ' ' + str(y) + ' 1* ' + fluid + ' /\n'

        for x, name, y, z1, z2, skin, rw in zip(all_well_xs, all_well_names, all_well_ys,
                                              all_well_z1s, all_well_z2s, skin, rw):
            if template == 2 or template == 4: name = f'"{name}"'
            if horizontal:
                self.compdat = name + ' ' + str(x) + ' ' + str(y) + ' ' + str(z2) + ' ' + str(z2) + ' OPEN	1*	1* ' + str(rw) +  ' 1* ' + str(skin) + ' 1* Y /\n' 
                for i in range(y+1, y_stop[0]+1):
                    self.compdat += name + ' ' + str(x) + ' ' + str(i) + ' ' + str(z2) + ' ' + str(z2) + ' OPEN	1*	1* ' + str(rw) +  ' 1* ' + str(skin) + ' 1* Y /\n'
            else:
                for i in range(0, len(all_well_xs)):
                    self.compdat += name + ' ' + str(all_well_xs[i]) + ' ' + str(all_well_ys[i]) + ' ' + str(all_well_z1s[i]) + ' ' + str(all_well_z2s[i]) + ' OPEN	1*	1*	' + str(rw) +  ' 1* ' + str(skin) + ' /\n'

        for prod, rezim, q_oil, prod_bhp in zip(prod_names, rezim, prod_q_oil, prod_bhp):
            if template == 2 or template == 4: prod = f'"{prod}"'
            self.wconprod += prod + ' OPEN ' + rezim + ' ' + str(q_oil) + ' 4* ' + str(prod_bhp) + ' /'

        self.wconinje = ''
        if not only_prod:
            for inj, inj_bhp in zip(inj_names, inj_bhp):
                if template == 2 or template == 4: inj = f'"{inj}"'
                self.wconinje += inj + ' WAT OPEN BHP ' + str(inj_bhp) + ' 1* /'

        self.template = template # выбираем шаблон data файла для различных симуляторов
        # templates: 1-opm; 2-ecl (в разработке)

        # Моделирование ГРП:
        self.grp = grp
        self.nz_grp = nz_grp
        self.xs_start_grp = xs_start_grp
        self.xs_stop_grp = xs_stop_grp
        self.ys_grp = ys_grp
        self.k_grp = k_grp

        self.grp_word = ''
        if grp == True:
            self.grp_word = 'EQUALS \n'
            self.grp_word += f"'PERMX' {k_grp} {xs_start_grp} {xs_stop_grp} {ys_grp} {ys_grp} {nz_grp} {nz_grp} /\n"
            self.grp_word += f"'PERMY' {k_grp} {xs_start_grp} {xs_stop_grp} {ys_grp} {ys_grp} {nz_grp} {nz_grp} /\n"
            self.grp_word += '/'

        # Моделирование потерь на трение
        self.welsegs = ''
        self.compsegs = ''
        if roughness:
            for name in zip(all_well_names):
                self.welsegs += f'"{name}" {tops_depth+z2} 0.0 1* INC HF- /\n'
                self.welsegs += f'2 {y_stop[0]}  1 1 {y_stop[0]+all_well_z2s[0]} 0 {rw*2} {roughness} /\n'
                self.compsegs += f'"{name}" /\n'
                self.compsegs +=  f' {all_well_xs[0]} {all_well_ys[0]} {all_well_z2s[0]} 1 1 1* Y {all_well_ys[-1]} /\n'

        # переменные для расчета
        self.result_df = None
        self.fig = None
        self.fig_npv = None
        self.dir = None
        self.result_data = None
        self.create_data_file()

        # переменный для рисования
        self.grid_dx = nx
        self.grid_dy = ny
     
    def create_data_file(self):
        if self.template == 1:
            template_name = 'templates/opm.DATA'
        elif self.template == 2:
            template_name = 'templates/ecl.DATA'
        elif self.template == 3:
            template_name = 'templates/opm_multphase.DATA'
        elif self.template == 4:
            template_name = 'templates/ecl_multphase.DATA'
        env = Environment(loader=FileSystemLoader(''))
        template = env.get_template(template_name)
        self.result_data = template.render(DIMENS=self.dimens, START=self.start_date,
            DX=self.dx, DY=self.dy, DZ=self.dz, TOP_BOX=self.top_box, TOPS=self.tops_depth, PORO_BOX=self.poro_box, PORO=self.por,
            PERMX=self.permx, PERMY=self.permy, PERMZ=self.permz, ROCK=self.rock,  DENSITY=self.density,
            EQUIL=self.equil, WELSPECS=self.welspecs, COMPDAT=self.compdat,
            WCONPROD=self.wconprod, WCONINJE=self.wconinje, TSTEP=self.tstep, GRP=self.grp_word, WELSEGS=self.welsegs, COMPSEGS=self.compsegs)


    def create_model(self, name, result_name, keys):
        self.save_file(name=name)
        if self.template == 1 or self.template == 3:
            self.calculate_file(name)
            self.create_result(name=name, keys=keys)
            self.read_result(name=result_name)


    def save_file(self, name):
        with open('model_folder/'+ name + '.DATA', "w") as file:
            file.write(self.result_data)

    @staticmethod
    def calculate_file(name):
        os.system("mpirun flow model_folder/%s.DATA" % name)

    @staticmethod
    def create_result(name, keys):
        summary = EclSum('model_folder/%s.DATA' % name)
        dates = summary.dates
        results = []
        all_keys = []

        if keys is None:
            keys = ["WOPR:*"]

        for key in keys:
            key_all_wells = summary.keys(key)
            all_keys = all_keys + list(key_all_wells)

        for key in all_keys:
            results.append(list(summary.numpy_vector(key)))

        if len(results) == 0:
            return print('Результаты из модели не загрузились. Файл с результатами не был создан')

        result_df = pd.DataFrame(data=np.array(results).T, index=dates, columns=all_keys)
        result_df.index.name = 'Time'
        result_df.to_csv('csv_folder/%s.csv' % name)
        print('%s_RESULT.csv is created' % name)


    def read_result(self, name):
        self.result_df = pd.read_csv('csv_folder/%s.csv' % name, parse_dates=[0], index_col=[0])
        print('%s.csv is read' % name)
        

    # метод построения графика по заданному параметру
    def summ_plot(self, parameters=None, mode='lines', x_axis=None, y_axis=None, title=None, name=None):
        directory = "csv_folder/"
        self.fig = go.Figure()
        files = [f for f in os.listdir(directory)]
        files.sort(key=lambda x:int(x.split('.')[1]))
        i = 0
        for file in files:
            df = pd.read_csv('csv_folder/%s' % file, parse_dates=[0], index_col=[0])
            # i = int(file.split('.')[1])
            self.fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[parameters[0]],
                    mode=mode,
                    name='Модель:' + name[i]))
            i += 1
        if x_axis is None:
            x_axis = 'Дата'
        if y_axis is None:
            y_axis = ''
        if title is None:
            title = 'Динамика показателей по моделям'
        self.fig.update_xaxes(tickformat='%d.%m.%y')
        self.fig.update_layout(title=go.layout.Title(text=title),
                               xaxis_title=x_axis,
                               yaxis_title=y_axis)
        iplot(self.fig)


    def npv_plot(self, name=None, l_list=None):
        directory = "csv_folder/"
        npv_list = []
        model_list = []
        files = [f for f in os.listdir(directory)]
        files.sort(key=lambda x:int(x.split('.')[1]))
        for file in files:
            df = pd.read_csv('csv_folder/%s' % file, parse_dates=[0], index_col=[0])
            i = int(file.split('.')[1])
            npv_ = self.npv_method(df, l_list[i])
            npv_list.append(npv_)
            model_list.append(f'Модель: {name[i]}')
        colors = ['lightslategray',] * 10
        #colors[6] = 'crimson'
        data = [go.Bar(
            x = model_list,
            y = npv_list,
            marker_color=colors)]
        self.fig_npv = go.Figure(data=data)
        self.fig_npv.update_layout(title='NPV по моделям')
        self.fig_npv.update_yaxes(type="log")
        iplot(self.fig_npv)

    # метод для отображения графика с оптимальной плотностью сетки
    def summ_plot_plotn(self, parameters=None, A=None, name=None, l=None, x_axis=None, y_axis=None, title=None):
        directory = "csv_folder/"
        npv_list = []
        model_list = []
        self.fig = go.Figure()
        files = [f for f in os.listdir(directory)]
        files.sort(key=lambda x:int(x.split('.')[1]))
        i = 0
        for file in files:
            df = pd.read_csv('csv_folder/%s' % file, parse_dates=[0], index_col=[0])
            j = int(file.split('.')[1])
            y = df[parameters[0]]/A[i]
            npv_ = self.npv_plotn_method(df, l, A[i])
            npv_list.append(npv_)
            model_list.append(f'Модель: {name[i]}')
            self.fig.add_trace(go.Scatter(
                x=df.index, y=y, mode='lines', name=f'Модель: {name[i]}'
            ))
            i += 1

        self.fig.update_xaxes(tickformat='%d.%m.%y')
        self.fig.update_layout(title=go.layout.Title(text=title),
                xaxis_title=x_axis,
                yaxis_title=y_axis)
        colors = ['lightslategray',] * 6
        colors[2] = 'crimson'
        data = [go.Bar(
            x = model_list,
            y = npv_list,
            marker_color=colors)]
        self.fig_npv = go.Figure(data=data)
        self.fig_npv.update_layout(title='NPV по моделям')
        iplot(self.fig_npv)
        iplot(self.fig)

    # методы расчета NPV для исследования скважин   
    def npv_method(self, df, l):
        ci = 170*10**6 # руб, капитальные затраты на строительство скважины c поверхностным обустройством;
        cap_l = 40000 # 3кк*73/3к=73к РУБ, стоимость 1 метра горизонтального ствола;
        p = 4500 # 62*73*6,3=28500 # руб/м3, net-baсk цена нефти за вычетом НПДИ и подготовку нефти; 
        opex = 10**6 # руб/год, операционные затраты на скважину;
        r = 0.12 # ставка дисконтирования;
        to = df.index[0]
        i = 0
        npv = -ci - l*cap_l
        j = 0
        q = 0
        for t in df.index:
            if abs((t - to).days) >= 365:
                i += 1
                to = t
                q = df['FOPT'][j] - q
                dcf = (q*p - opex)/(1 + r)**i
                npv += dcf 
            j += 1

        return round(npv, 0)


    def npv_plotn_method(self, df, l, A):
        ci = 170*10**6 # руб, капитальные затраты на строительство скважины c поверхностным обустройством;
        cap_l = 40000 # РУБ, стоимость 1 метра горизонтального ствола;
        p = 15000 # руб/м3, net-baсk цена нефти за вычетом НПДИ и подготовку нефти; 
        opex = 10**6 # руб/год, операционные затраты на скважину;
        r = 0.12 # ставка дисконтирования;
        to = df.index[0]
        i = 0
        npv = (-ci - l*cap_l)/A
        j = 0
        q = 0
        for t in df.index:
            if abs((t - to).days) >= 365:
                i += 1
                to = t
                q = df['FOPT'][j] - q
                dcf = (q*p - opex)/A/(1 + r)**i
                npv += dcf 
            j += 1

        return round(npv, 0)

    # метод для построения графика с оптимальным соотношнием сторон
    def sootn_plot(self, ls, par, k, h, mu, A):
        self.fig = go.Figure()
        x_opt = []
        y_opt = []
        ind = [-1, -1, -1, [-5, -5, -6, -5, -5, -6, -5, -5, -5, -5], [-5, -5, -5, -5, -6, -5], -1]
        x_apr, y_apr = [], []
        for i in range(0, 6):
            P = []
            directory = f"csv_folder/{i}"
            files = [f for f in os.listdir(directory)]
            files.sort(key=lambda x:int(x.split('.')[2]))
            j = 0
            for file in files:
                df = pd.read_csv(f'csv_folder/{i}/%s' % file, parse_dates=[0], index_col=[0])
                if i == 3 or i == 4:
                    val = k*h*(df['FPR'][ind[i][j]]-df['WBHP:P1'][ind[i][j]])/(df['WOPR:P1'][ind[i][j]]*mu)*10**5*86400
                else:    
                    val = k*h*(df['FPR'][ind[i]]-df['WBHP:P1'][ind[i]])/(df['WOPR:P1'][ind[i]]*mu)*10**5*86400
                val = val + h*(10)**0.5/2/3.14/400*(m.log(h*(10)**0.5/(2*3.14*0.5*0.156*(1+(10)**0.5)*m.sin(3.14/2)))+ 0) # 0 в конце это скин
                #print(df['WOPR:P1'])
                P.append(val)
                j += 1
            f_interp = interp1d(ls[i] , P, bounds_error=False)
            x_val = A[i]*10000/(A[i]*10000+160000)
            # print(x_val)
            # print(f_interp(x_val))
            if f_interp(x_val) > 0:
                x_apr.append(x_val)
                y_apr.append(f_interp(x_val))
            self.fig.add_trace(go.Scatter(
                x=[x_val], y=[f_interp(x_val)], mode='markers', name='Оптимальное соотн. для A/L^2 = ' + str(par[i])
            ))
            self.fig.add_trace(go.Scatter(
                x=ls[i],
                y=P,
                mode='lines',
                name='Параметр - A/L^2 = ' + str(par[i])))
        t = np.polyfit(x_apr, y_apr, 2)
        f = np.poly1d(t)
        print('Таким образом, для нашей системы оптимальное соотношение парктически идеально описывается следующим уравнением:')
        print(f)
        for i in range(0, 6):
            x_opt.append(A[i]*10000/(A[i]*10000+400**2))
            y_opt.append(f(x_opt[i]))
        # x_opt.append(200*10000/(200*10000+400**2))
        # y_opt.append(m.log(1+200*10000/(400**2))*3.5)
    
        self.fig.add_trace(go.Scatter(
                x=x_opt,
                y=y_opt,
                mode='lines',
                name='Оптимальное соотношение геометрических размеров'
            ))
        x_axis = 'Соотношение геометрических размеров области дренирования (ширина/длина)'
        y_axis = 'Безразмерный перепад давлений P*'
        title = 'Безразмерный перепад давления как функция соотн. геометрических размеров области дренирования'
        self.fig.update_layout(title=go.layout.Title(text=title),
                               xaxis_title=x_axis,
                               yaxis_title=y_axis)
        iplot(self.fig)

    # метод для формирования измельченной сетки
    # необходимо помнить, что размеры ячеек должны отличаться не более чем в 2 раза
    @staticmethod     
    def setcas(nx, lx, s, v):
        k = 1
        l = 0
        delta = lx*0.01
        while abs(l - lx) > delta:
            if k <= 2:
                k += 0.001
            else:
                print('Невозможно разбить сетку!')
                raise Exception() 
            n = round((nx - s) / 2)
            l = v*s + 2 * (v * k * (k ** n - 1) / (k - 1))
            if abs(l - lx) < delta:
                x = []
                for i in range(1, n + 1):
                    x.append(round(v * k ** i, 3))
                x = x[::-1] + [v] * s + x
                return x


    @staticmethod     
    def grid_df(filename, parametr, dx, dy):
        eclfiles = EclFiles(f'model_folder/{filename}.DATA')
        dates = 'last'
        dframe = grid.df(eclfiles, rstdates=dates)
        df = []
        k = dx
        for i in range(0, dy):
            ra = []
            for j in range(0, dx):
                ra.append(dframe[str(parametr)].iloc[j+i*k])
            df.append(ra)
        
        return df


    def grid_plot(self, filename, parametr, title=''):
        df = self.grid_df(filename, parametr, self.grid_dx, self.grid_dy)
        self.fig = px.imshow(df, labels=dict(x="nx", y="ny", color=title))
        iplot(self.fig)


    # # метод для построения графиков на основе summary (тест)
    # def summary_df():
    #     from ecl2df import summary, EclFiles

    #     eclfiles = EclFiles("model_folder/TEST_MODEL_HORIZONTAL.0.DATA")
    #     dframe = summary.df(eclfiles, column_keys="FOPT", time_index="monthly") # daily
