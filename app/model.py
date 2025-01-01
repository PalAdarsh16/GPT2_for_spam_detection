from peewee import Model, SqliteDatabase, CharField, TextField
db = SqliteDatabase('checker.db')
class SpamCheckModel(Model):
    text = TextField()
    
    output = TextField(null = True)

    class Meta:
        database = db

db.create_tables([SpamCheckModel])        