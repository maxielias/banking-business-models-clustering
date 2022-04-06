from dbfread import DBF

table = DBF('IND.dbf', load=True)

print(table.records[0])
