import pandas as pd
import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')


def split_train_test(df):
    """
    df_test = df[df['report'] >= 863]
    df_train = df[df['report'] < 863]
    df_test.to_csv("data\df_test_tokens_labeled_iob.csv", index = False)
    df_train.to_csv("data\df_train_tokens_labeled_iob.csv", index = False)
    """
    
    # Same reports used in llms
    test_report_index = [540, 541, 546, 550, 552, 554, 556, 558, 559, 577, 583, 589, 591, 595, 598, 613, 615, 616, 618, 627, 
                 629, 637, 640, 646, 650, 659, 660, 662, 665, 666, 673, 677, 684, 691, 693, 694, 697, 699, 701, 702, 
                 703, 706, 707, 712, 713, 719, 720, 725, 726, 727, 731, 734, 741, 744, 747, 749, 751, 752, 753, 754, 
                 759, 760, 771, 774, 776, 792, 795, 797, 806, 811, 813, 818, 819, 820, 821, 822, 830, 832, 834, 836, 
                 839, 847, 848, 851, 853, 864, 865, 867, 871, 873, 874, 875, 877, 880, 881, 885, 886, 893, 896, 900]
    
    df_test_llms = df[df['report'].isin(test_report_index)]
    df_train_llms = df[~df['report'].isin(test_report_index)]

    #df_test_llms.to_csv("data/df_test_llms.csv", index = False)
    #df_train_llms.to_csv("data/df_train_llms.csv", index = False)
    
    return df_train_llms, df_test_llms

def tokens_and_tags_single_to_grouped(df, phase):
    # Converter as colunas para strings
    df['token'] = df['token'].astype(str)
    df['iob_label'] = df['iob_label'].astype(str)

    # Agrupar as palavras e os rótulos por número do laudo
    grouped = df.groupby('report').agg({'token': ' '.join, 'iob_label': ' '.join}).reset_index()

    # Renomear as colunas
    grouped.columns = ['report', 'text', 'iob_labels']

    # Salvar o resultado em um novo arquivo CSV
    #file_name = 'data/df_'+ phase +'_full_sentence.csv'
    #grouped.to_csv(file_name, index=False)
    
    return grouped

def split_text_string(string):
    #tokens = wordpunct_tokenize(string)
    tokens = word_tokenize(string, language="portuguese")

    n = len(tokens)
    div_size = n // 4
    divided_parts = [tokens[i*div_size:(i+1)*div_size] for i in range(4)]
    return divided_parts

def split_labels_string(string):
    tokens = word_tokenize(string)
    n = len(tokens)
    div_size = n // 4
    divided_parts = [tokens[i*div_size:(i+1)*div_size] for i in range(4)]
    return divided_parts

def split_text_df(df, phase):
    # Dividindo as strings em 4 partes
    df['text_parts'] = df['text'].apply(split_text_string)
    df['iob_label_parts'] = df['iob_labels'].apply(split_labels_string)

    # Criando o novo dataframe com as strings divididas
    new_df = pd.DataFrame(columns=['report', 'text', 'iob_labels'])

    for _, row in df.iterrows():
        for i in range(4):
            report = row['report']
            sentence_part = ' '.join(row['text_parts'][i])
            tags_part = ' '.join(row['iob_label_parts'][i])
            aux_dict = {'report': report, 'text': sentence_part, 'iob_labels' : tags_part}
            new_df_row = pd.DataFrame([aux_dict])
            new_df = pd.concat([new_df, new_df_row], ignore_index=True)

    file_name = 'data/'+'df_'+phase+'_full_sentences_divide_by_four.csv'
    
    new_df.to_csv(file_name, index=False)

def main():
    df = pd.read_csv('data/df_tokens_labeled_iob.csv')            
    train_df, test_df = split_train_test(df)
    df_all_reports_full_sentence = tokens_and_tags_single_to_grouped(df, 'all_resports')
    df_all_reports_full_sentence.to_csv('data/df_all_reports_full_sentence.csv', index=False)

    #train_df = pd.read_csv('data/df_train_llms.csv')
    #test_df = pd.read_csv('data/df_test_llms.csv')
    train_df_full_sentence = tokens_and_tags_single_to_grouped(train_df, 'train_llms')
    test_df_full_sentence = tokens_and_tags_single_to_grouped(test_df, 'test_llms')

    #train_df_full = pd.read_csv('data/df_train_llms_full_sentence.csv')
    #test_df_full = pd.read_csv('data/df_test_llms_full_sentence.csv')
    split_text_df(train_df_full_sentence, 'train_llms')
    split_text_df(test_df_full_sentence, 'test_llms')

if __name__ == "__main__":
    main()

