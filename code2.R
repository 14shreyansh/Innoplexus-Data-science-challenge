train_data<-read.csv('Train_data.csv')
test_data<-read.csv('Test_data.csv')

# function to remove special characters
removeSpecialChars <- function(x) gsub("[^a-zA-Z0-9 ]", " ", x)
# remove special characters
train_data$text <- sapply(train_data$text, removeSpecialChars)
test_data$text <- sapply(test_data$text, removeSpecialChars)

fix.contractions <- function(doc) {
  # "won't" is a special case as it does not expand to "wo not"
  doc <- gsub("won't", "will not", doc)
  doc <- gsub("can't", "can not", doc)
  doc <- gsub("n't", " not", doc)
  doc <- gsub("'ll", " will", doc)
  doc <- gsub("'re", " are", doc)
  doc <- gsub("'ve", " have", doc)
  doc <- gsub("'m", " am", doc)
  doc <- gsub("'d", " would", doc)
  # 's could be 'is' or could be possessive: it has no expansion
  doc <- gsub("'s", "", doc)
  return(doc)
}

# fix (expand) contractions
train_data$text <- sapply(train_data$text, fix.contractions)
test_data$text <- sapply(test_data$text, fix.contractions)

library(tm)
train_corpus <- VCorpus(VectorSource(c(train_data$text,test_data$text)))
##Removing Punctuation
train_corpus <- tm_map(train_corpus, content_transformer(removePunctuation))
##Removing numbers
train_corpus <- tm_map(train_corpus, removeNumbers)
##Converting to lower case
train_corpus <- tm_map(train_corpus, content_transformer(tolower))
##Removing stop words
train_corpus <- tm_map(train_corpus, content_transformer(removeWords), stopwords())
##Stemming
train_corpus <- tm_map(train_corpus, stemDocument)
##Whitespace
train_corpus <- tm_map(train_corpus, stripWhitespace)

# Create Document Term Matrix
dtm_train <- DocumentTermMatrix(train_corpus)

train_corpus <- removeSparseTerms(dtm_train, 0.99)

dtm_train_matrix <- as.matrix(train_corpus)

important_words_df <- as.data.frame(dtm_train_matrix)
colnames(important_words_df) <- make.names(colnames(important_words_df))

important_words_train_df <- head(important_words_df, nrow(train_data))
important_words_test_df <- tail(important_words_df, nrow(test_data))

train_data_words_df <- cbind(train_data, important_words_train_df)
test_data_words_df <- cbind(test_data, important_words_test_df)

# Get rid of the original Text field
train_data_words_df$text <- NULL
test_data_words_df$text <- NULL

train_data_words_df <- train_data_words_df[, !duplicated(colnames(train_data_words_df))]
test_data_words_df <- test_data_words_df[, !duplicated(colnames(test_data_words_df))]

train_data_words_df$sentiment<-as.factor(train_data_words_df$sentiment)

test_data_words_df$sentiment <- NA

test_data_words_df$sentiment<-as.factor(test_data_words_df$sentiment)

train_data_words_df<-train_data_words_df[,-c(1,2)]
test_data_words_df<-test_data_words_df[,-c(1,2)]

table(train_data_words_df$sentiment)

library(e1071)
classifier = svm(formula = sentiment ~ .,
                 data = train_data_words_df,
                 type = 'C-classification',
                 kernel = 'linear',
                 cost=10,
                 gamma=15
                 )


y_pred = predict(classifier, newdata = test_data_words_df[-1900])

result<-as.data.frame(y_pred)
summary(result)

result<-cbind(result,test_data$unique_hash)

write.csv(result,'result.csv')

svm_tune <- tune(svm, train.x=train_data_words_df[-1], train.y=train_data_words_df$sentiment, 
                 kernel="linear", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))
