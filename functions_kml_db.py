 # created by Timur
import pandas as pd
import sqlalchemy
import geopy.distance
import psycopg2
import pandas.io.sql as sqlio
from bs4 import BeautifulSoup
import os
import json
from contextlib import closing
from psycopg2.extras import DictCursor
from psycopg2 import sql
from tables import table
from dotenv import load_dotenv
from pathlib import Path
import functions_geo
import kml_parser_new_format


def load_connect_params():
    load_dotenv()
    env_path = Path('.')/'.env'
    load_dotenv(dotenv_path=env_path)
    conn_params = {
        'user': os.getenv("PG_USER"),
        'password': os.getenv("PG_PASSWORD"),
        'host': os.getenv("PG_HOST"),
        'port': os.getenv("PG_PORT"),
        'dbname': os.getenv("PG_DBNAME")
    }
    return conn_params


def dbConnect_by_psycopg2():  # подключение к бд по psycorp2
    try:
        conn_params = load_connect_params()
        connect = psycopg2.connect(**conn_params)
        return connect
    except Exception as exc:
        return {'Code': 0, 'Msg': exc}


# функция для создания таблиц
def query_by_psycopg2(sql_command): # запросы для создания таблиц хранятся отдельно в виде словаря, {'tab1_name':'sql_command',...}
    #prepare posgresql
    connect = dbConnect_by_psycopg2()
    cursor = connect.cursor()
    try:
        cursor.execute(sql_command)
        connect.commit()
        return {'Code':1, 'Msg': 'table(s) created'}
    except Exception as exc:
        return {'Code':0,'Msg':exc}


# функция подключения к бд по SQLAlchemy
def dbConnect():
    connect_params = load_connect_params()
    connection_string = 'postgresql+psycopg2://'
    connection_string += connect_params['user'] + ':' + connect_params['password'] + '@'
    connection_string += connect_params['host'] + ':' + connect_params['port'] + '/' + connect_params['dbname']
    try:
        engine = sqlalchemy.create_engine(connection_string)
        return engine
    except Exception as exc:
        return exc


# функция получения всех линий
def get_all_lines_data():
    engine = dbConnect()
    query = "SELECT voltageclass, district, substation, dname,feeder, iddzo, filename FROM lines"
    all_lines_df = pd.read_sql_query(query, engine)
    return all_lines_df


# функция получения данных линии по его id
def get_line_data(id_):
    engine = dbConnect()
    query = "SELECT voltageclass,district,substation,dname,feeder,filename FROM lines WHERE iddzo='" + id_ +"'"
    line_data_df = pd.read_sql_query(query, engine)
    return line_data_df


def get_all_lines_data_with_pyllons_fname():
    df_all_lines_data = get_all_lines_data()
    df_all_lines_data[['filename']] = ''
    df_pylons_kml_filename = get_pylons_kml_filename()
    for index, row in df_pylons_kml_filename.iterrows():
        df_all_lines_data.loc[(df_all_lines_data.iddzo == row['iddzo']), 'filename'] = row['filename']
    # pd.DataFrame(df_all_lines_data).to_csv('moesc/lines_data_with_pyllons_fname.csv')
    return df_all_lines_data


def get_pylons_kml_filename():
    engine = dbConnect()
    query = "SELECT lines.voltageclass, lines.district," \
            "lines.substation, lines.dname, lines.feeder," \
            "lines.iddzo, pylons.filename " \
            "FROM pylons, lines " \
            "WHERE pylons.idlinedzo1=lines.iddzo" \
            " GROUP BY lines.iddzo, pylons.filename"
    df_pyllons_fname = pd.read_sql_query(query, engine)

    return df_pyllons_fname


# функция получения данных линии по его id
def get_feeder_data(feeder_id):
    if feeder_id == '':
        df_pillars = pd.read_csv('test_gazstroi_coords.csv')
        return df_pillars

    engine = dbConnect()
    query = "SELECT voltageclass,district,substation,dname,feeder,iddzo,filename FROM lines WHERE feeder='" + feeder_id +"'"
    df_feeder_main = pd.read_sql_query(query, engine)
    df_feeder_main.to_csv('test_df_feeder_main.csv', index=False)
    # voltageclass = list(set(df_feeder_main['voltageclass']))[0]
    # district = list(set(df_feeder_main['district']))[0]
    # substation = list(set(df_feeder_main['substation']))[0]
    # filename = list(set(df_feeder_main['filename']))[0]
    # dname = list(set(df_feeder_main['dname']))[0]
    # line_ids = set(df_feeder_main['iddzo'])

    df_feeder_data = pd.DataFrame()

    for line_id, group in df_feeder_main.groupby('iddzo'):
        df_ = pylons_lat_lon_df(line_id)
        df_['iddzo'] = line_id

        voltageclass = list(group['voltageclass'])[0]
        district = list(group['district'])[0]
        substation = list(group['substation'])[0]
        filename = list(group['filename'])[0]
        dname = list(group['dname'])[0]

        df_['voltageclass'] = voltageclass
        df_['district'] = district
        df_['substation'] = substation
        df_['filename'] = filename
        df_['dname'] = dname

        df_feeder_data = df_feeder_data.append(df_, ignore_index=True)

    # df_feeder_data['voltageclass'] = voltageclass
    # df_feeder_data['district'] = district
    # df_feeder_data['substation'] = substation
    # df_feeder_data['filename'] = filename
    # df_feeder_data['dname'] = dname
    # df_feeder_data['feeder_id'] = feeder_id

    return df_feeder_data


# функция получения списка id линий
def get_LinesIddzo():
    engine = dbConnect()
    query = "SELECT iddzo FROM lines"
    linesIddzo_dict = (pd.read_sql_query(query ,engine)).to_dict()
    linesIddzo = list()
    keys = linesIddzo_dict['iddzo'].keys()
    for key in keys:
        linesIddzo.append(linesIddzo_dict['iddzo'][key])

    return linesIddzo # возвращаем список id линий


def pylons_lat_lon_df(IdDZO): # вернет в виде массива кортежей, вида [(lat,lon),(lat,lon),...]
    engine = dbConnect()
    query = "SELECT idlinedzo1,iddzo,lat,lon FROM pylons WHERE idlinedzo1 = '" + IdDZO + "' ORDER BY iddzo"
    df = pd.read_sql_query(query, engine)
    df = df[['lat','lon']]
    #df = df.sort_values(by='iddzo')
    #line_coords = [(lat, lon) for lat, lon in zip(df['lat'], df['lon'])]
    return df


def calcDistanceByLineId(): # вычисляем длину линии по его id и возвращаем словарь {'iddzo':idDZO, 'lenght' : lenght}
    linesIddzo = get_LinesIddzo() # получили список линий
    lineLenghtById = list()
    for lineId in linesIddzo:
        line_coords = pylons_lat_lon_df(lineId) # получили координаты столбов
        line_coords = line_coords.to_numpy()
        length = 0
        for pylon_coord1, pylon_coord2 in zip(line_coords, line_coords[1:]):
            length +=functions_geo.get_geo_distance(pylon_coord1, pylon_coord2)
        lineLenghtById.append({'idDZO':lineId, 'length':length})
    return lineLenghtById


# запись в бд длины линий по id
def insertLinesLengthById():
    try:
        linesLength = calcDistanceByLineId()
        for line in linesLength:
            sql = "UPDATE lines set linelength =" + str(line['length']) + "   WHERE iddzo='" + line['idDZO'] + "'"
            query_by_psycopg2(sql)
        return {'Code':1,'Msg':'SUCCESSFULL'}
    except Exception as exc:
        return {'Code':0,'Msg':exc}


# удаление линии и столбов по id линии
def delete_line(line_id):
    connect = dbConnect_by_psycopg2()
    cursor = connect.cursor()
    sql = "DELETE FROM pylons WHERE idlinedzo1='"+ line_id +"'; DELETE FROM lines WHERE iddzo='"+ line_id +"'"
    try:
        cursor.execute(sql)
        connect.commit()
        cursor.close()
        connect.close()
    except Exception as exc:
        return exc

# функция вывода пролетов
def get_span_names(idDZO):
    engine = dbConnect()
    query = "SELECT iddzo,number1 FROM pylons WHERE idlinedzo1 = '" + idDZO + "' ORDER BY iddzo"
    df = pd.read_sql_query(query, engine)
    spans = df['number1'].to_numpy()
    span_list = list()
    for pylon1, pylon2 in zip(spans,spans[1:]):
        span_list.append(str(pylon1)+'-'+str(pylon2))
    return span_list


# парсер kml файла с последующим добавлением в бд
def kml_parser(kml_file):
    try:
        #prepare posgresql
        connect = dbConnect_by_psycopg2()
        cursor = connect.cursor()
    except Exception as exc:
        return {'code': 0, 'Msg': exc}

    parser = kml_parser_new_format.parser(kml_file)
    kml_type = parser[0]
    parsed_data = parser[1]
    if kml_type == 'lines':
        for data in parsed_data:
            # записываем в базу линии парсим из массива словарей сначала столбы таблицы и соответсвующие им значения
            sql = "INSERT INTO lines (" + ", ".join(data.keys()) + ") VALUES (" + ", ".join(
                ["%(" + c + ")s" for c in data]) + ") ON CONFLICT (iddzo) DO NOTHING"
            try:
                cursor.execute(sql, data)
                connect.commit()
            except Exception as exc:
                return {'code': 0, 'Msg': exc}
        return {'code': 1, 'Msg': 'INSERTED SUCCESSFULL'}

    elif kml_type == 'pylons':
        for data in parsed_data:
            # записываем в базу линии парсим из массива словарей сначала столбы таблицы и соответсвующие им значения
            sql = "INSERT INTO pylons (" + ", ".join(data.keys()) + ") VALUES (" + ", ".join(
                ["%(" + c + ")s" for c in data]) + ")  ON CONFLICT (iddzo) DO NOTHING"
            try:
                cursor.execute(sql, data)
                connect.commit()
            except Exception as exc:
                return {'code': 0, 'Msg': exc}
        return {'code': 1, 'Msg': 'INSERTED SUCCESSFULL'}
    else:
        return {'code': 0, 'Msg': 'BAD FILE'}


def get_feeder_id_by_line_id(line_id):
    line_data_df = get_line_data(line_id)
    feeder_id = list(line_data_df['feeder'])[0]
    return feeder_id
