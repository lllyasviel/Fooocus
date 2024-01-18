import sqlite3

# Function to connect to the SQLite database
def connect_to_db(db_path):
    return sqlite3.connect(db_path)

# Function to get categories and text chunks from the database
def get_categories_and_texts(db_path):
    conn = connect_to_db(db_path)
    cursor = conn.cursor()

    # Get categories
    cursor.execute("SELECT Oid, Title FROM TextClipCategory")
    categories = cursor.fetchall()

    # Get text chunks for each category
    texts = {}
    for category in categories:
        cat_id = category[0]
        cursor.execute("SELECT TextChunk FROM TextClip WHERE CategoryOid = ?", (cat_id,))
        texts[cat_id] = cursor.fetchall()

    conn.close()
    return categories, texts

# Function to save a text chunk to a category
def save_text_to_category(db_path, text, category_oid):
    conn = connect_to_db(db_path)
    cursor = conn.cursor()

    cursor.execute("INSERT INTO TextClip (TextChunk, CategoryOid) VALUES (?, ?)", (text, category_oid))
    conn.commit()
    conn.close()

# Example usage
# categories, texts = get_categories_and_texts('/path/to/DataAF.sqlite')
# save_text_to_category('/path/to/DataAF.sqlite', 'Example Text', 1)
