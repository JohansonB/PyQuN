# MSLoader: A Generic DataLoader for ModelSet

**MSLoader** is a flexible and extensible data loader for the `ModelSet` datatype. It provides an abstract framework for users to implement custom loaders that parse structured data files into a hierarchical `ModelSet` representation.

---

## **Overview**

To create a custom data loader:

- **Extend `DataLoader`**:
  - Implement the `read_file` method to define your own data loading logic.

- **Extend `MSLoader`**:
  - Use this class as a base for data loaders specifically tailored to the `ModelSet` structure.
  - Includes an abstract workflow for parsing models, elements, and attributes.

The framework also provides a ready-to-use implementation for loading CSV files.

---

## **Built-in CSV Loader**

An implementation for loading ModelSets from CSV files is included

---
