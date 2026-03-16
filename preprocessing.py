import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def main():

    # Load dataset
    df = pd.read_csv("data/HR-Employee-Attrition-Data.csv")
    print("Dataset Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nDataset Info:")
    print(df.info())

    print("\nStatistical Summary:")
    print(df.describe())

    print("\nFirst Rows:")
    print(df.head())

    # Attrition Distribution
    print("\nAttrition Value Counts:")
    print(df['Attrition'].value_counts())

    print("\nAttrition Percentage:")
    print(
        df['Attrition']
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .astype(str) + '%'
    )

    # Plot Attrition Distribution
    df['Attrition'].value_counts().plot(
        kind='bar',
        color=['steelblue', 'salmon'],
        edgecolor='black'
    )

    plt.title('Attrition Class Distribution')
    plt.xlabel('Attrition')
    plt.ylabel('Number of Employees')
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.show()

    # Convert target variable
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    print("\nAttrition after conversion:")
    print(df['Attrition'].value_counts())

    # Drop unnecessary columns
    cols_to_drop = [
        'EmployeeCount',
        'EmployeeNumber',
        'Over18',
        'StandardHours'
    ]

    cols_to_drop = [col for col in cols_to_drop if col in df.columns]

    df.drop(columns=cols_to_drop, inplace=True)

    print("\nDropped columns:", cols_to_drop)
    print("New shape:", df.shape)

    # Categorical encoding
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    print("\nCategorical columns to encode:")
    print(categorical_cols)

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    print("\nShape after encoding:", df.shape)

    # Correlation analysis
    correlation_with_target = df.corr()['Attrition'].drop('Attrition')

    correlation_sorted = correlation_with_target.abs().sort_values(ascending=False)

    print("\nTop 15 features correlated with Attrition:")
    print(correlation_sorted.head(15).round(4))

    # Correlation graph
    correlation_sorted.head(15).plot(
        kind='bar',
        color='steelblue',
        edgecolor='black'
    )

    plt.title('Top 15 Features Correlated with Attrition')
    plt.ylabel('Absolute Correlation')
    plt.xlabel('Feature')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

    # Select top features
    top_10_features = correlation_sorted.head(10).index.tolist()

    print("\nSelected top 10 features:")
    for i, feat in enumerate(top_10_features, 1):
        print(f"{i}. {feat}")

    # Create model dataset
    df_model = df[top_10_features + ['Attrition']].copy()

    print("\nFinal model dataset shape:", df_model.shape)

    # Feature target split
    X = df_model.drop('Attrition', axis=1)
    y = df_model['Attrition']

    print("\nFeatures (X) shape:", X.shape)
    print("Target (y) shape:", y.shape)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining set size:", X_train.shape)
    print("Testing set size:", X_test.shape)

    # Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nScaling complete")

    # Apply SMOTE
    smote = SMOTE(random_state=42)

    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_scaled,
        y_train
    )

    print("\nSMOTE applied successfully")

    # Class distribution
    before = pd.Series(y_train).value_counts().sort_index()
    after = pd.Series(y_train_resampled).value_counts().sort_index()

    print("\nClass Distribution BEFORE SMOTE")
    print(before)

    print("\nClass Distribution AFTER SMOTE")
    print(after)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(
        ['No (0)', 'Yes (1)'],
        [before.get(0, 0), before.get(1, 0)],
        color=['steelblue', 'salmon']
    )

    axes[0].set_title('Before SMOTE')

    axes[1].bar(
        ['No (0)', 'Yes (1)'],
        [after.get(0, 0), after.get(1, 0)],
        color=['steelblue', 'salmon']
    )

    axes[1].set_title('After SMOTE')

    plt.suptitle('Training Set Class Distribution')

    plt.tight_layout()

    plt.show()

    # Save processed datasets
    X_train_resampled_df = pd.DataFrame(X_train_resampled, columns=X.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

    y_train_resampled_df = pd.DataFrame(y_train_resampled, columns=["Attrition"])
    y_test_df = pd.DataFrame(y_test, columns=["Attrition"])

    X_train_resampled_df.to_csv("data/X_train.csv", index=False)
    X_test_scaled_df.to_csv("data/X_test.csv", index=False)

    y_train_resampled_df.to_csv("data/y_train.csv", index=False)
    y_test_df.to_csv("data/y_test.csv", index=False)

    print("\nProcessed datasets saved successfully.")


if __name__ == "__main__":
    main()